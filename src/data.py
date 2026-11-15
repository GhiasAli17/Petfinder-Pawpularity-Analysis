# src/data.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def build_transforms(img_size, aug_type, train=True):
    if aug_type == "lite":
        if train:
            t = [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            t = [transforms.Resize((img_size, img_size))]
    elif aug_type == "strong":
        if train:
            t = [
                 
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
                transforms.RandomAffine(20, shear=10, scale=(0.8, 1.2)),
                transforms.RandomPerspective(0.2, p=0.5),
               
            ]
        else:
            t = [transforms.Resize((img_size, img_size))]
    else:
        raise ValueError("Unknown aug_type: %s" % aug_type)
    # this is added for both train and valid and have default values as per ImageNet
    t += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(t)

class JointImageMaskTransform:
    """Apply identical geometric transformations to image and pet mask."""

    def __init__(self, img_size, aug_type, train=True):
        if aug_type not in ("lite", "strong"):
            raise ValueError(f"Unknown aug_type: {aug_type}")

        self.img_size = img_size
        self.aug_type = aug_type
        self.train = train
        self.jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)

    def __call__(self, image, mask):
        import torch
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode as I

        size = [self.img_size, self.img_size]

        image = TF.resize(
            image, size, interpolation=I.BILINEAR, antialias=True
        )
        mask = TF.resize(mask, size, interpolation=I.NEAREST)

        if self.train:
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if self.aug_type == "strong":
                angle = transforms.RandomRotation.get_params(
                    [-30.0, 30.0]
                )
                image = TF.rotate(
                    image, angle, interpolation=I.NEAREST, fill=0
                )
                mask = TF.rotate(
                    mask, angle, interpolation=I.NEAREST, fill=0
                )

                # Colour changes apply only to the photograph.
                image = self.jitter(image)

                params = transforms.RandomAffine.get_params(
                    degrees=[-20.0, 20.0],
                    translate=None,
                    scale_ranges=[0.8, 1.2],
                    shears=[-10.0, 10.0],
                    img_size=TF.get_image_size(image),
                )
                image = TF.affine(
                    image, *params, interpolation=I.NEAREST, fill=0
                )
                mask = TF.affine(
                    mask, *params, interpolation=I.NEAREST, fill=0
                )

                if torch.rand(1).item() < 0.5:
                    width, height = image.size
                    start, end = transforms.RandomPerspective.get_params(
                        width, height, distortion_scale=0.2
                    )
                    image = TF.perspective(
                        image, start, end,
                        interpolation=I.BILINEAR, fill=0
                    )
                    mask = TF.perspective(
                        mask, start, end,
                        interpolation=I.NEAREST, fill=0
                    )

        image = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image, TF.to_tensor(mask)

class ImageOnlyDataset(Dataset):
    def __init__(
        self, df, img_folder, transform=None,
        aux_tasks=None, binary_aux_tasks=None,
        joint_transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform
        self.aux_tasks = aux_tasks or []
        self.binary_aux_tasks = binary_aux_tasks or []
        self.joint_transform = joint_transform

        bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        self.has_bbox = all(c in df.columns for c in bbox_cols)

        if joint_transform is not None:
            required = bbox_cols + ["yolo_conf"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing saliency data columns: {missing}"
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_folder, row["Id"] + ".jpg")

        with Image.open(path) as source:
            image = source.convert("RGB")

        width, height = image.size
        aux = {}

        if self.has_bbox:
            bbox = np.zeros(4, dtype=np.float32)

            # Preserve your existing detection-confidence threshold.
            if float(row.get("yolo_conf", 0.0)) > 0.3:
                bbox = np.array(
                    [
                        row["bbox_x1"], row["bbox_y1"],
                        row["bbox_x2"], row["bbox_y2"],
                    ],
                    dtype=np.float32,
                )
                if not np.isfinite(bbox).all():
                    raise ValueError(f"Invalid bbox for {row['Id']}")

            aux["pet_bbox"] = bbox

        if self.joint_transform is not None:
            # Build the mask at the ORIGINAL photograph dimensions.
            raster = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = aux["pet_bbox"]

            x1 = int(np.clip(x1, 0, width))
            x2 = int(np.clip(x2, 0, width))
            y1 = int(np.clip(y1, 0, height))
            y2 = int(np.clip(y2, 0, height))

            if x2 > x1 and y2 > y1:
                raster[y1:y2, x1:x2] = 255

            image, aux["pet_mask"] = self.joint_transform(
                image, Image.fromarray(raster)
            )

        elif self.transform is not None:
            image = self.transform(image)

        if "brisque" in self.aux_tasks:
            aux["brisque"] = np.float32(row["BRISQUE"])

        if "visibility_ratio" in self.aux_tasks:
            aux["visibility_ratio"] = np.float32(
                row["visibility_ratio"]
            )

        for task in self.binary_aux_tasks:
            if task in row.index:
                aux[task] = np.float32(row[task])

        return image, row["Pawpularity"], aux
    
# class ImageOnlyDataset(Dataset):
#     def __init__(self, df, img_folder, transform=None, aux_tasks=None, binary_aux_tasks=None):
#         self.df = df.reset_index(drop=True)
#         self.img_folder = img_folder
#         self.transform = transform
#         self.aux_tasks = aux_tasks or []
#         self.binary_aux_tasks = binary_aux_tasks or [] # binary aux tasks from metadata not pseudo-labels

#         # check if bbox columns exist for saliency loss
#         bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
#         self.has_bbox = all(c in df.columns for c in bbox_cols)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_id = row["Id"]
#         path = os.path.join(self.img_folder, img_id + ".jpg")
#         img = Image.open(path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         y = row["Pawpularity"]
#         aux = {}
#         if "brisque" in self.aux_tasks:
#             aux["brisque"] = np.float32(row["BRISQUE"])
#         if "visibility_ratio" in self.aux_tasks:
#             aux["visibility_ratio"] = np.float32(row["visibility_ratio"])

#         for task in self.binary_aux_tasks:
#             if task in row.index:
#                 aux[task] = np.float32(row[task])

                
#         if self.has_bbox:
#             conf = float(row.get("yolo_conf", 0.0))
#             if conf > 0.3:
#                 aux["pet_bbox"] = np.array([
#                     row["bbox_x1"], row["bbox_y1"],
#                     row["bbox_x2"], row["bbox_y2"]
#                 ], dtype=np.float32)
#             else:
#                 aux["pet_bbox"] = np.zeros(4, dtype=np.float32)    

#         return img, y, aux
        # return img, y


class ImageTabDataset(Dataset):
    def __init__(self, df, img_folder, tab_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.tab_cols = tab_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]
        path = os.path.join(self.img_folder, img_id + ".jpg")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = row[self.tab_cols].values.astype(np.float32)
        y = row["Pawpularity"]
        return img, tab, y
