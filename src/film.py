import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation:
    y = gamma * x + beta, with gamma, beta predicted from a conditioning input Metadata.
    """
    def __init__(self, cond_dim, feature_dim):
        super().__init__()

        # Linear layer to predict per-channel gamma and beta from meta.
        # Input:  (B, cond_dim)
        # Output: (B, feature_dim=C)
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc  = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        """
        x: (B, C, H, W)
        cond: (B, tabular_dim)
        """
        gamma = self.gamma_fc(cond)  # (B, C)
        beta  = self.beta_fc(cond)   # (B, C)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)

        return gamma * x + beta # (B, C, H, W) FiLM-modulated


## ********* PART 1********************** ##
# FiLMInternalModulation: apply FiLM on internal backbone blocks (in-place modulation).

class FiLMInternalModulation(nn.Module):
    """
    - film_start_idx:  FiLM starting from this block index.
    - apply_to_all_after: if True, FiLM will be applied to all blocks >= film_start_idx;
                          if False, only at film_start_idx.
    - use_bn_affine: if False,  BN affine(gamma and beta) will be disabled so BN only normalizes
                     and FiLM provides the affine(gamma and beta).
    """
    def __init__(
        self,
        backbone_name: str,
        img_size: int,
        tab_input_dim: int, # dimension of tabular input features ,e.g 12
        tab_hidden: int = 64, # tab hidden dim, e.g. 64
        film_start_idx: int = 5, # first backbone block index where FiLM is applied, e.g. block/stage 5 of EfficientNet Architecture
        apply_to_all_after: bool = True, #apply FiLM to all blocks >= film_start_idx
        head_hidden: int = 256, # head dimension
        pretrained: bool = True, # finetune from pretrained backbone
        use_bn_affine: bool = True,  # whether to use gamma and beta from Batch Norm, or disable and rely on FiLM for affine transformation(gamma/beta)
        freeze_backbone: bool = False, # whether to freeze the backbone weights (except FiLM layers) or finetune them
    ):
        super().__init__()

        extra_kwargs = {}
        if "swin" in backbone_name or "vit" in backbone_name:
            extra_kwargs["img_size"] = img_size
            extra_kwargs["dynamic_img_pad"] = True

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            **extra_kwargs,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False



        self.film_start_idx = film_start_idx  
        self.apply_to_all_after = apply_to_all_after
        self.use_bn_affine = use_bn_affine

        # to disable BN affine (gamma/beta) in blocks where FiLM is used.
        # iterating through that block and set affine false a
        if not self.use_bn_affine:
            if hasattr(self.backbone, "blocks"):
                for idx, block in enumerate(self.backbone.blocks):
                    use_film_here = (
                        idx == self.film_start_idx
                        or (self.apply_to_all_after and idx > self.film_start_idx)
                    )
                    if not use_film_here:
                        continue
                    
                    # For all BatchNorm2d inside this block, make them "norm only".

                    for m in block.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.affine = False
                            if m.weight is not None:
                                nn.init.ones_(m.weight) #gamma=1
                                m.weight.requires_grad = False
                            if m.bias is not None:
                                nn.init.zeros_(m.bias) #beta=0
                                m.bias.requires_grad = False

        # Tabular encoder to project raw tabular feature to projected dim for FiLM conditioning
        self.tab_enc = nn.Sequential(
            nn.Linear(tab_input_dim, tab_hidden),
            nn.ReLU(),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(),
        )
        self.film_cond_dim = tab_hidden

        # --- PREBUILD FiLM modules here using a dummy forward ---
        # We want one FiLM layer for each backbone block where FiLM is applied.
        # But different blocks can have different channel sizes C (e.g., 24, 40, 80, ...),
        # so we must know x.shape[1] for each block to create FiLM(cond_dim, feature_dim=C).
        # To discover these C values, we run a single dummy forward pass and
        # create the FiLM modules on-the-fly, then reuse them during training.

        self.film_layers = nn.ModuleDict()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size) #dummy image
            x = self.backbone.conv_stem(dummy)
            x = self.backbone.bn1(x)
            if getattr(self.backbone, "act1", None) is not None:
                x = self.backbone.act1(x)
             # Iteration over all backbone blocks 
            for idx, block in enumerate(self.backbone.blocks):
                x = block(x)  # x: (1, C_block, H_block, W_block)
                use_film_here = (
                    idx == self.film_start_idx
                    or (self.apply_to_all_after and idx > self.film_start_idx)
                )
                if use_film_here:
                    c = x.shape[1] # Feature channel dimension at this point
                    #  cond_dim: dimension of tabular conditioning vector (self.film_cond_dim)
                    #  feature_dim: number of channels C at this block
                    # This FiLM will learn gamma/beta per channel to modulate x.
                    self.film_layers[str(idx)] = FiLM(
                        cond_dim=self.film_cond_dim,
                        feature_dim=c,
                    )
        # --------------------------------------------------------

        self.num_features = self.backbone.num_features
        self.global_pool = self.backbone.global_pool

        self.head = nn.Sequential(
            nn.Linear(self.num_features, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def _get_or_create_film(self, idx: int, x: torch.Tensor):
        # to get the film layer/block/stage
        return self.film_layers[str(idx)]

    def _forward_features_with_film(self, x: torch.Tensor, tab_feat: torch.Tensor):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        if getattr(self.backbone, "act1", None) is not None:
            x = self.backbone.act1(x)
                
        # loopiong over blocks and apply FiLM as per idx.

        for idx, block in enumerate(self.backbone.blocks):
            x = block(x)
            use_film_here = (
                idx == self.film_start_idx
                or (self.apply_to_all_after and idx > self.film_start_idx)
            )
            if use_film_here:
                #  the FiLM module corresponding to this block index.
                film = self._get_or_create_film(idx, x)
                #  FiLM to modulate the current feature map x using tab_feat
                x = film(x, tab_feat)

        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        if getattr(self.backbone, "act2", None) is not None:
            x = self.backbone.act2(x)

        return x

    def forward(self, img: torch.Tensor, tab: torch.Tensor):
        tab_feat = self.tab_enc(tab)
        x = self._forward_features_with_film(img, tab_feat)
        x = self.global_pool(x)
        out = self.head(x)
        return out

    
    
# ********* PART 2********************** 
# FiLMExternalModulation:  FiLM after backbone feature extraction, 
# with extra FiLM-ed ResBlocks added on top (not in-place modulation of backbone blocks).    


class FiLMedResBlock(nn.Module):
    """ Architecuter of single FiLM-ed ResBlock (single FiLM per block):

      x -> Conv(1x1) -> ReLU -> Conv(3x3) -> BN(affine=False) -> FiLM -> ReLU -> +skip
                             |                                                  |
                             |                                                  ^
                              ->------------------------------------------------|
                                Skip connection/Residual form
                             
    """
    def __init__(self, channels=128, cond_dim=128):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels, affine=False)  # BN normalize only

        self.film  = FiLM(cond_dim, channels)   # single FiLM per block
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, cond):
        # x: (B, C=128, H=14, W=14)
        out = self.conv1(x)
        out = self.relu1(out)

        residual = out  # skip from after first ReLU

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film(out, cond)
        out = self.relu2(out)

        out = out + residual # (B, 128, 14, 14), residual/skip addition
        return out


class FiLMedResStack(nn.Module):
    """
    Stack of FiLMedResBlockExact, all at Bx128x14x14.
    """
    def __init__(self, num_blocks=4, channels=128, cond_dim=128):
        super().__init__()
        #  list of FiLMedResBlock modules.
        # Each block takes (x: (B, C, 14, 14), cond: (B, cond_dim))
        # and returns (B, C, 14, 14).
        self.blocks = nn.ModuleList([
            FiLMedResBlock(channels=channels, cond_dim=cond_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x

class FiLMClassifier(nn.Module):
    """
     classifier:

      Input: 128x14x14 
      1x1 conv: 128->512
      Global max-pool
      MLP: 512->1024->out_dim
    """
    def __init__(self, in_channels=128, out_dim=1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 512, kernel_size=1, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        # x: (B, 128, 14, 14) 
        x = self.conv1x1(x)           # (B, 512, 14, 14)
        x = F.max_pool2d(x, kernel_size=x.shape[-2:])  # global max-pool -> (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # Flatten (B, 512)
        out = self.mlp(x)             # (B, out_dim)
        return out




class BackboneFeatureExtractor(nn.Module):
    """
    EfficientNet -> 128 x 14 x 14 feature map for FiLM external modulation.

     conv_stem + bn1 + act1 + blocks, then a 3x3 conv to 128 channels,
    then adaptive pooling to 14x14.
    """
    def __init__(self, backbone_name="efficientnet_b1", pretrained=True, freeze=False):
        super().__init__()

        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        self.conv_stem = backbone.conv_stem
        self.bn1       = backbone.bn1
        self.act1      = getattr(backbone, "act1", None)
        self.blocks    = backbone.blocks

        # --- determine C_in after blocks using a dummy pass ---
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)  #  Dummy input just to infer output channel dimension;
            x = self.conv_stem(dummy)           # (1, C_stem, H/2, W/2)
            x = self.bn1(x)
            if self.act1 is not None:
                x = self.act1(x)
            for block in self.blocks:
                x = block(x)                    # after last block: (1, C_in, H', W')
            in_c = x.shape[1]                   # e.g. 320 for B1, 448 for B4

       #project to 128 channels for FiLM block 
        self.project_to_128 = nn.Conv2d(
            in_c, 128, kernel_size=3, padding=1, bias=False
        )

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.act1 is not None:
            x = self.act1(x)
        for block in self.blocks:
            x = block(x)              # (B, C_in, H', W')
    
      # Project channels to 128 with a 3x3 conv.

        x = self.project_to_128(x)            # (B, 128, H', W')
        if x.shape[-2] != 14 or x.shape[-1] != 14:
            x = F.adaptive_avg_pool2d(x, output_size=(14, 14))  # (B, 128, 14, 14)
        return x
    





class FiLMExternalModulation(nn.Module):
    """
    Paper-like FiLM visual pipeline with tab features as conditioning:

       BackboneFeatureExtractor)
      -> FiLMedResStackExact (4 blocks, 128x14x14)
      -> Final classifier (1x1 conv to 512, max-pool, MLP)
      -> regression output (1 unit for Pawpularity)
    """
    def __init__(
        self,
        backbone_name: str,  #  "efficientnet_b1"
        tab_input_dim: int,
        tab_hidden: int = 128,
        num_film_blocks: int = 4,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

    
        if backbone_name == "efficientnet_b1" or backbone_name == "efficientnet_b4":
            self.backbone = BackboneFeatureExtractor(
                backbone_name=backbone_name,
                pretrained=pretrained_backbone,
                freeze=freeze_backbone,
            )
        else:
            raise ValueError(f"Unknown backbone_name: {backbone_name}")

        # Tabular encoder produces conditioning vector (plays the role of GRU embedding)
        self.tab_enc = nn.Sequential(
            nn.Linear(tab_input_dim, tab_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(inplace=True),
        )

        # 4 FiLMed residual blocks on 128x14x14
        self.film_stack = FiLMedResStack(
            num_blocks=num_film_blocks,
            channels=128,
            cond_dim=tab_hidden,
        )

        # classifier 
        self.classifier = FiLMClassifier(in_channels=128, out_dim=1)

    def forward(self, img, tab):
        x = self.backbone(img)      # (B, 128, 14, 14)
        cond = self.tab_enc(tab)    # (B, tab_hidden)
        x = self.film_stack(x, cond)  # (B, 128, 14, 14) FiLM-ed features
        # final classifier head
        out = self.classifier(x)    # (B, 1)
        return out