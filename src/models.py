# src/models.py
import torch
import torch.nn as nn
import timm

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, out_dim=1):
        super(TabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_vision_backbone(name, img_size, mode):
    """
    img_size is used for ViT/Swin-type models that have a fixed patch embedding size. Because by default swin tinny accepts 224 input size
    """
    extra_kwargs = {}
    # flexible size for Swin.
    if "swin" in name or "vit" in name:
        extra_kwargs["img_size"] = img_size
        extra_kwargs["dynamic_img_pad"] = True

    if mode == "feature":
        model = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
            **extra_kwargs,
        )
    elif mode == "regression":
        model = timm.create_model(
            name,
            pretrained=True,
            num_classes=1,
            **extra_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


class EarlyFusionNet(nn.Module):
    """
    Vision backbone (features only) + tabular MLP encoder + fusion head.
    head_type: "linear" or "mlp".
    """
    def __init__(
        self,
        backbone_name,
        img_size,
        tab_input_dim,
        tab_hidden=64,
        fusion_hidden=256,
        head_type="mlp",
        pretrained=True,
    ):
        super(EarlyFusionNet, self).__init__()

        self.backbone_name = backbone_name  

        extra_kwargs = {}
        # flexible size for Swin / ViT
        if "swin" in backbone_name or "vit" in backbone_name:
            extra_kwargs["img_size"] = img_size
            extra_kwargs["dynamic_img_pad"] = True

        self.img_model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            **extra_kwargs,
        )
        img_out_dim = self.img_model.num_features

        self.tab_enc = nn.Sequential(
            nn.Linear(tab_input_dim, tab_hidden),
            nn.ReLU(),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(),
        )
        tab_out_dim = tab_hidden

        fusion_in = img_out_dim + tab_out_dim
        if head_type == "linear":
            self.head = nn.Linear(fusion_in, 1)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(fusion_in, fusion_hidden),
                nn.ReLU(),
                nn.Linear(fusion_hidden, 1),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, img, tab):
        img_feat = self.img_model(img)
        tab_feat = self.tab_enc(tab)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        out = self.head(fused)
        return out


# class EarlyFusionNet(nn.Module):
#     """
#     Vision backbone (features only) + tabular MLP encoder + fusion head.
#     head_type: "linear" or "mlp".
#     """
#     def __init__(
#         self,
#         backbone_name,
#         img_size,
#         tab_input_dim,
#         tab_hidden=64,
#         fusion_hidden=256,
#         head_type="mlp",
#         pretrained=True,
#     ):
#         super(EarlyFusionNet, self).__init__()

#         extra_kwargs = {}
#         # flexible size for Swin.
#         if "swin" in backbone_name or "vit" in backbone_name:
#             extra_kwargs["img_size"] = img_size
#             extra_kwargs["dynamic_img_pad"] = True

#         self.img_model = timm.create_model(
#             backbone_name,
#             pretrained=pretrained,
#             num_classes=0,
#             **extra_kwargs,
#         )
#         img_out_dim = self.img_model.num_features

#         self.tab_enc = nn.Sequential(
#             nn.Linear(tab_input_dim, tab_hidden),
#             nn.ReLU(),
#             nn.Linear(tab_hidden, tab_hidden),
#             nn.ReLU(),
#         )
#         tab_out_dim = tab_hidden

#         fusion_in = img_out_dim + tab_out_dim
#         if head_type == "linear":
#             self.head = nn.Linear(fusion_in, 1)
#         elif head_type == "mlp":
#             self.head = nn.Sequential(
#                 nn.Linear(fusion_in, fusion_hidden),
#                 nn.ReLU(),
#                 nn.Linear(fusion_hidden, 1),
#             )
#         else:
#             raise ValueError("Unknown head_type: %s" % head_type)

#     def forward(self, img, tab):
#         img_feat = self.img_model(img)
#         tab_feat = self.tab_enc(tab)
#         fused = torch.cat([img_feat, tab_feat], dim=1)
#         out = self.head(fused)
#         return out
