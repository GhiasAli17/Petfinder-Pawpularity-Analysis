import torch.nn as nn
import shap
import joblib
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import numpy as np
from PIL import Image
import torch
import os

def get_last_conv_layer(model):
    """
    Return the last Conv2d layer in a CNN backbone (e.g., EfficientNet in timm).
    """
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layers found for Grad-CAM.")
    return conv_layers[-1]




def compute_cnn_gradcam(backbone, pil_image, img_size, target_layer, device="cuda"):
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img_tensor = val_tf(pil_image).unsqueeze(0)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device).eval()

    cam = GradCAM(
        model=backbone,
        target_layers=[target_layer],
    )
    grayscale_cam = cam(input_tensor=img_tensor.to(device))[0]  # [Hc, Wc]

    # overlay
    rgb = np.array(pil_image).astype(np.float32) / 255.0
    h, w, _ = rgb.shape
    cam_img = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
    cam_img = cam_img.resize((w, h), resample=Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0
    vis = show_cam_on_image(rgb, cam_resized, use_rgb=True)
    return vis



def lgbm_feature_importance(models_dir, tab_cols, n_folds, out_path):
    importances = np.zeros(len(tab_cols))
    for fold in range(1, n_folds + 1):
        m = joblib.load(os.path.join(models_dir, f"lgbm_fold{fold}.pkl"))
        importances += m.feature_importances_
    importances /= n_folds
    fi_df = pd.DataFrame({"feature": tab_cols, "importance": importances})
    fi_df.sort_values("importance", ascending=False).to_csv(out_path, index=False)
    return fi_df


def lgbm_shap_global(model_path, X_sample, out_dir, prefix="exp1"):
    os.makedirs(out_dir, exist_ok=True)
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    np.save(os.path.join(out_dir, f"{prefix}_shap_values.npy"), shap_values)
    np.save(os.path.join(out_dir, f"{prefix}_shap_data.npy"), X_sample.values)
    return shap_values