import torch
def saliency_loss(spatial_feat, pet_masks, return_stats=False):
    """
    Feature-activation regularization using aligned pet masks.

    spatial_feat: BHWC, or BNC with a square spatial grid.
    pet_masks:   Bx1xHxW, aligned with the transformed input images.
    """
    if spatial_feat.dim() == 3:
        batch, tokens, channels = spatial_feat.shape
        side = int(tokens ** 0.5)

        if side * side != tokens:
            raise ValueError("Expected a square spatial token grid.")

        spatial_feat = spatial_feat.reshape(
            batch, side, side, channels
        )

    if spatial_feat.dim() != 4:
        raise ValueError(
            f"Expected BHWC features, got {tuple(spatial_feat.shape)}"
        )

    # Preserve your existing channel-mean activation definition.
    activation = spatial_feat.float().mean(dim=-1)

    if not torch.isfinite(activation).all():
        raise FloatingPointError("Nonfinite spatial activations.")

    minimum = activation.amin(dim=(1, 2), keepdim=True)
    maximum = activation.amax(dim=(1, 2), keepdim=True)

    # Correct min–max normalization.
    normalized = (
        (activation - minimum) / (maximum - minimum + 1e-9)
    )

    masks = pet_masks.to(
        device=activation.device, dtype=torch.float32
    )

    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    if (
        masks.dim() != 4
        or masks.shape[0] != activation.shape[0]
        or masks.shape[1] != 1
    ):
        raise ValueError("Expected masks with shape Bx1xHxW.")

    if not torch.isfinite(masks).all():
        raise ValueError("Nonfinite pet mask.")

    # Convert the aligned input mask to the actual feature-grid size.
    masks = torch.nn.functional.interpolate(
        masks,
        size=activation.shape[1:],
        mode="nearest",
    )[:, 0]

    masks = (masks > 0.5).to(normalized.dtype)

    inside_count = masks.sum(dim=(1, 2))
    outside_count = (1.0 - masks).sum(dim=(1, 2))

    # Both regions must exist for an inside/outside comparison.
    valid = (inside_count > 0) & (outside_count > 0)

    inside = (
        (normalized * masks).sum(dim=(1, 2))
        / inside_count.clamp_min(1.0)
    )
    outside = (
        (normalized * (1.0 - masks)).sum(dim=(1, 2))
        / outside_count.clamp_min(1.0)
    )

    if valid.any():
        loss = ((1.0 - inside) + outside)[valid].mean()
    else:
        loss = spatial_feat.sum() * 0.0

    stats = {
        "saliency_valid": int(valid.sum().item()),
        "saliency_empty": int((inside_count == 0).sum().item()),
        "saliency_full": int((outside_count == 0).sum().item()),
    }

    return (loss, stats) if return_stats else loss

# def saliency_loss(spatial_feat, pet_bboxes, img_size=384):
#     """
#     Spatial feature activation regularization loss.
#     Uses last Swin block feature map as feature activation proxy.
#     Penalizes activation outside the pet bbox.

#     spatial_feat : (B, H, W, C) or (B, N, C)
#     pet_bboxes   : (B, 4) float tensor (x1, y1, x2, y2) pixel coords
#                    all-zero bbox = no detection = sample skipped
#     img_size     : input image size
#     """
#     if spatial_feat.dim() == 4:
#         B, H, W, C = spatial_feat.shape
#     elif spatial_feat.dim() == 3:
#         B, N, C = spatial_feat.shape
#         H = W = int(N ** 0.5)
#         spatial_feat = spatial_feat.reshape(B, H, W, C)
#     else:
#         raise ValueError(f"Unexpected spatial_feat shape: {spatial_feat.shape}")

#     #  feature activation map: mean over channels -> (B, H, W)
#     # average over channels -> (B, H, W), then normalize each sample to [0, 1]
#     # this gives a spatial heatmap of where the backbone is "attending"
#     featureActMap = spatial_feat.mean(dim=-1)
#     featureActMap_flat = featureActMap.flatten(1)
#     featureActMap = featureActMap - featureActMap_flat.min(dim=1)[0].view(B, 1, 1)
#     featureActMap = featureActMap / (featureActMap_flat.max(dim=1)[0].view(B, 1, 1) + 1e-9)

#     # scale = H / img_size
#     scale_x = W / img_size
#     scale_y = H / img_size
#     losses = []

#     for b in range(B):
#         x1, y1, x2, y2 = pet_bboxes[b]

#         # skip images with no YOLO detection
#         if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
#             continue


#         fx1 = int(max(0, float(x1 * scale_x)))
#         fy1 = int(max(0, float(y1 * scale_y)))
#         fx2 = int(min(W, float(x2 * scale_x)))
#         fy2 = int(min(H, float(y2 * scale_y)))

#         #skip degenerate boxes (can happen if pet is very small)
#         if fx2 <= fx1 or fy2 <= fy1:
#             continue

#         # binary pet mask at feature map resolution 
#         # 1.0 inside pet bbox, 0.0 outside
#         mask = torch.zeros(H, W, device=spatial_feat.device)
#         mask[fy1:fy2, fx1:fx2] = 1.0

#          # compute inside/outside feature activation fractions
#         # inside  = mean feature activation inside pet bbox  ( HIGH -> close to 1)
#         # outside = mean feature activation outside pet bbox ( LOW  -> close to 0)
#         inside  = (featureActMap[b] * mask).sum() / (mask.sum() + 1e-9)
#         outside = (featureActMap[b] * (1 - mask)).sum() / ((1 - mask).sum() + 1e-9)
#         losses.append((1 - inside) + outside)

#     # if no valid bboxes in this batch, return zero loss 
#     # spatial_feat.sum() * 0.0 keeps the tensor in the computation graph
#     # so gradients can still flow (unlike torch.tensor(0.0, requires_grad=True))
#     if len(losses) == 0:
#         # return torch.tensor(0.0, device=spatial_feat.device, requires_grad=True)
#         return spatial_feat.sum() * 0.0
#     # average loss over all valid images in the batch
#     return torch.stack(losses).mean()


def focal_mse_loss(pred, target, gamma=1.0):
    """
    Focal MSE loss for Pawpularity regression.
    Gives more weight to hard samples (large prediction errors).
    gamma=0 -> standard MSE
    gamma=1 -> error-weighted MSE (default)
    gamma=2 -> strongly emphasizes hard samples
    
    pred   : (B, 1) predicted Pawpularity
    target : (B, 1) true Pawpularity
    """
    error  = (pred - target) ** 2
    weight = torch.abs(pred.detach() - target) ** gamma
    return (weight * error).mean()