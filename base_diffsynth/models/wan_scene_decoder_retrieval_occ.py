# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import math
import torch.nn.functional as F

try:
    import xformers.ops as xops
except ImportError:
    xops = None # Handle fallback in code

def init_weights(module, std=0.02):
    """Initialize weights for linear and embedding layers.
    
    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)



# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)



class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    """
    
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        bias=False,
        dropout=0.0,
        activation=nn.GELU,
        mlp_dim=None,
    ):
        """
        Args:
            dim: Input dimension
            mlp_ratio: Multiplier for hidden dimension
            bias: Whether to use bias in linear layers
            dropout: Dropout probability
            activation: Activation function
            mlp_dim: Optional explicit hidden dimension (overrides mlp_ratio)
        """
        super().__init__()
        hidden_dim = mlp_dim if mlp_dim is not None else int(dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            activation(),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)



class QK_Norm_SelfAttention(nn.Module):
    """
    Self-attention with optional Q-K normalization.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            qkv_bias: Whether to use bias in QKV projection
            fc_bias: Whether to use bias in output projection
            attn_dropout: Dropout probability for attention weights
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K normalization
        We use flash attention V2 for efficiency.
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_bias=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attn_bias: Optional attention bias mask
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        
        # Apply qk normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        x = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.attn_dropout if self.training else 0.0,
            op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
        )
        
        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        return x




class SubsetAttention(nn.Module):
    """Attention that can attend to subsets of queries or keys/values."""
    
    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        attn_dropout=0.0,
        fc_bias=False,
        fc_dropout=0.0,
        use_qk_norm=False
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            qkv_bias: Whether to use bias in QKV projection
            attn_dropout: Dropout probability for attention weights
            fc_bias: Whether to use bias in output projection
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K normalization
        We use flash attention V2 for efficiency.
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm

        # Projections
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, subset_kv_size=None, subset_q_size=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            subset_kv_size: If provided, only attend to tokens after this index in KV
            subset_q_size: If provided, only compute attention for queries up to this index
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Only one subset parameter can be provided
        assert not (subset_kv_size is not None and subset_q_size is not None), \
            "Only one of subset_kv_size or subset_q_size can be provided"

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Handle subset attention cases
        if subset_kv_size is not None and subset_kv_size < k.shape[1]:
            # Attend to subset of key/value tokens
            k_subset = k[:, subset_kv_size:, :, :].contiguous()
            v_subset = v[:, subset_kv_size:, :, :].contiguous()
            
            x = xops.memory_efficient_attention(
                q, k_subset, v_subset,
                attn_bias=None,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        elif subset_q_size is not None and subset_q_size < q.shape[1]:
            # Only compute attention for subset of query tokens
            q_subset = q[:, :subset_q_size, :, :].contiguous()
            
            x = xops.memory_efficient_attention(
                q_subset, k, v,
                attn_bias=None,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            # Regular attention for all tokens
            x = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=None,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        
        x = rearrange(x, "b l nh dh -> b l (nh dh)")

        # Final projection
        x = self.attn_fc_dropout(self.fc(x))
        
        return x




class QK_Norm_TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-normalization architecture.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    """

    def __init__(
        self,
        dim,
        head_dim,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, bias=ln_bias)
        self.attn = QK_Norm_SelfAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )

        self.norm2 = nn.LayerNorm(dim, bias=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class ProcessDataPlucker(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item())
        w_orig = int(2 * cx.max().item())
        if h is None or w is None:
            h, w = h_orig, w_orig

        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d
    
    @torch.no_grad()
    def forward(self, data_batch, im_H, im_W, compute_rays=True):
        data_batch["image_h_w"] = (im_H, im_W)
        if compute_rays:
            c2w = data_batch["c2w"]
            fxfycxcy = data_batch["fxfycxcy"]
            image_height, image_width = data_batch["image_h_w"]

            ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, device=data_batch["c2w"].device)
            data_batch["ray_o"], data_batch["ray_d"] = ray_o, ray_d
        return data_batch

class ConvConfidenceHead(nn.Module):
    def __init__(self, in_dim, patch_h=256//8, patch_w=256//8):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.conv_net.apply(init_weights)

    def forward(self, x):
        b, n, d = x.shape
        h, w = self.patch_h, self.patch_w
        
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w).contiguous() 
        out = self.conv_net(x)
        
        return rearrange(out, 'b 1 h w -> b (h w) 1').contiguous()

class SceneDecoderOnlyOccAnalysis(nn.Module):
    def __init__(self, load_path=None, exit_layer=6):
        super().__init__()
        self.process_data = ProcessDataPlucker()

        self.input_pose_in_channels = 9
        self.target_pose_in_channels = 6
        self.patch_size = 8
        self.transformer_dim = 768
        self.transformer_d_head = 64
        self.n_layer = 24
        self.grad_checkpoint_every = 1

        self.image_size_H = 256
        self.image_size_W = 256

        self.exit_layer = exit_layer
        print(f"Exit layer: {self.exit_layer}")

        self.use_qk_norm = True
        self.special_init = True
        self.depth_init = True

        self._init_tokenizers()
        self._init_transformer()

        # Initialize Confidence Head (trainable)
        self.confidence_head = ConvConfidenceHead(
            in_dim=self.transformer_dim
        )

        if load_path is not None:
            self.load_ckpt(load_path)

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.input_pose_in_channels,
            patch_size = self.patch_size,
            d_model = self.transformer_dim
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.target_pose_in_channels,
            patch_size = self.patch_size,
            d_model = self.transformer_dim
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.transformer_dim, bias=False),
            nn.Linear(
                self.transformer_dim,
                (self.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)

    def _init_transformer(self):
        """Initialize transformer blocks"""
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                self.transformer_dim, self.transformer_d_head, use_qk_norm=self.use_qk_norm
            ) for _ in range(self.n_layer)
        ]
        
        # Apply special initialization if configured
        if self.special_init:
            for idx, block in enumerate(self.transformer_blocks):
                if self.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * self.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)
                
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(self.transformer_dim, bias=False)

    def pass_layers(self, input_tokens, start_layer=0, end_layer=None, gradient_checkpoint=False, checkpoint_every=1):
        """
        Args:
            input_tokens: 输入的 tokens
            start_layer: 开始执行的层索引
            end_layer: 结束执行的层索引（不包含）
        """
        num_layers = len(self.transformer_blocks)
        if end_layer is None:
            end_layer = num_layers
            
        for i in range(start_layer, end_layer):
            layer = self.transformer_blocks[i]
            if gradient_checkpoint and (i % checkpoint_every == 0):
                def custom_forward(x):
                    return layer(x)
                input_tokens = torch.utils.checkpoint.checkpoint(custom_forward, input_tokens, use_reentrant=False)
            else:
                input_tokens = layer(input_tokens)
                
        return input_tokens

    # def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
    #     """
    #     Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
    #     Args:
    #         input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
    #             The input tokens to process through the transformer blocks.
    #         gradient_checkpoint: bool, default False
    #             Whether to use gradient checkpointing to save memory during training.
    #         checkpoint_every: int, default 1 
    #             Number of transformer layers to group together for gradient checkpointing.
    #             Only used when gradient_checkpoint=True.
                
    #     Returns:
    #         Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
    #             The processed tokens after passing through all transformer blocks.
    #     """
    #     num_layers = len(self.transformer_blocks)
        
    #     if not gradient_checkpoint:
    #         # Standard forward pass through all layers
    #         for layer in self.transformer_blocks:
    #             input_tokens = layer(input_tokens)
    #         return input_tokens
            
    #     # Gradient checkpointing enabled - process layers in groups
    #     def _process_layer_group(tokens, start_idx, end_idx):
    #         """Helper to process a group of consecutive layers."""
    #         for idx in range(start_idx, end_idx):
    #             tokens = self.transformer_blocks[idx](tokens)
    #         return tokens
            
    #     # Process layer groups with gradient checkpointing
    #     for start_idx in range(0, num_layers, checkpoint_every):
    #         end_idx = min(start_idx + checkpoint_every, num_layers)
    #         input_tokens = torch.utils.checkpoint.checkpoint(
    #             _process_layer_group,
    #             input_tokens,
    #             start_idx,
    #             end_idx,
    #             use_reentrant=False
    #         )
            
    #     return input_tokens

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)

    def forward(self, input_data_batch, target_data_batch, ret_imgs=False):
        im_H, im_W = self.image_size_H, self.image_size_W

        val_input = self.process_data(input_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)
        val_target = self.process_data(target_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)

        # 1. Input embedding
        posed_input_images = self.get_posed_input(
            images=val_input['image'], ray_o=val_input['ray_o'], ray_d=val_input['ray_d']
        )
        b, v_input, c, h, w = posed_input_images.size()
        
        input_img_tokens = self.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)

        # 2. Target embedding
        target_pose_cond = self.get_posed_input(
            ray_o=val_target['ray_o'], ray_d=val_target['ray_d']
        )
        b_tgt, v_target, c_tgt, h_tgt, w_tgt = target_pose_cond.size()
        
        # Tokenize target poses: [b*v_target, n_patches, d]
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)

        # Repeat input tokens for each target view
        # [b, v_input*n_patches, d] -> [b*v_target, v_input*n_patches, d]
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        
        mid_tokens = self.pass_layers(
            concat_img_tokens, 
            start_layer=0, 
            end_layer=self.exit_layer, 
            gradient_checkpoint=True, 
            checkpoint_every=self.grad_checkpoint_every
        )
        
        _, mid_target_tokens = mid_tokens.split([v_input * n_patches, n_patches], dim=1)
        conf_log_var = self.confidence_head(mid_target_tokens) 

        final_tokens = self.pass_layers(
            mid_tokens, 
            start_layer=self.exit_layer,
            gradient_checkpoint=True, 
            checkpoint_every=self.grad_checkpoint_every
        )
        
        _, final_target_tokens = final_tokens.split([v_input * n_patches, n_patches], dim=1)
        rendered_images = self.image_token_decoder(final_target_tokens)

        height, width = self.image_size_H, self.image_size_W

        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // self.patch_size, 
            w=width // self.patch_size, 
            p1=self.patch_size, 
            p2=self.patch_size, 
            c=3
        )

        conf_log_var = rearrange(conf_log_var, '(b v) (h w) 1 -> (b v) 1 h w', 
                                 b=b, v=v_target, h=height // self.patch_size, w=width // self.patch_size)
        # conf_log_var_vis = F.interpolate(conf_log_var, size=(height, width), mode='bilinear', align_corners=False)

        nvs_ret = {
            "rendered_images": rendered_images,
            "conf_log_var": -conf_log_var,
            # "conf_log_var": -conf_log_var_vis,
        }
        return nvs_ret

    def calc_conf(self, input_data_batch, target_data_batch):
        im_H, im_W = self.image_size_H, self.image_size_W

        val_input = self.process_data(input_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)
        val_target = self.process_data(target_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)

        # 1. Input embedding
        posed_input_images = self.get_posed_input(
            images=val_input['image'], ray_o=val_input['ray_o'], ray_d=val_input['ray_d']
        )
        b, v_input, c, h, w = posed_input_images.size()
        
        input_img_tokens = self.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)

        # 2. Target embedding
        target_pose_cond = self.get_posed_input(
            ray_o=val_target['ray_o'], ray_d=val_target['ray_d']
        )
        b_tgt, v_target, c_tgt, h_tgt, w_tgt = target_pose_cond.size()
        
        # Tokenize target poses: [b*v_target, n_patches, d]
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)

        # Repeat input tokens for each target view
        # [b, v_input*n_patches, d] -> [b*v_target, v_input*n_patches, d]
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        
        mid_tokens = self.pass_layers(
            concat_img_tokens, 
            start_layer=0, 
            end_layer=self.exit_layer, 
            gradient_checkpoint=True, 
            checkpoint_every=self.grad_checkpoint_every
        )
        
        _, mid_target_tokens = mid_tokens.split([v_input * n_patches, n_patches], dim=1)
        conf_log_var = self.confidence_head(mid_target_tokens) 

        height, width = self.image_size_H, self.image_size_W

        conf_log_var = rearrange(conf_log_var, '(b v) (h w) 1 -> b v h w', 
                                 b=b, v=v_target, h=height // self.patch_size, w=width // self.patch_size)
        
        return -conf_log_var

    def calc_conf_batch(self, input_data_batch, target_data_batch):
        im_H, im_W = self.image_size_H, self.image_size_W

        val_input = self.process_data(input_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)
        val_target = self.process_data(target_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)

        # 1. Input embedding
        posed_input_images = self.get_posed_input(
            images=val_input['image'], ray_o=val_input['ray_o'], ray_d=val_input['ray_d']
        )
        b, v_input, c, h, w = posed_input_images.size()
        
        input_img_tokens = self.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)

        # 2. Target embedding
        target_pose_cond = self.get_posed_input(
            ray_o=val_target['ray_o'], ray_d=val_target['ray_d']
        )
        b_tgt, v_target, c_tgt, h_tgt, w_tgt = target_pose_cond.size()
        
        # Tokenize target poses: [b*v_target, n_patches, d]
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)

        # Repeat input tokens for each target view
        # [b, v_input*n_patches, d] -> [b*v_target, v_input*n_patches, d]
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        
        mid_tokens = self.pass_layers(
            concat_img_tokens, 
            start_layer=0, 
            end_layer=self.exit_layer, 
            gradient_checkpoint=True, 
            checkpoint_every=self.grad_checkpoint_every
        )
        
        _, mid_target_tokens = mid_tokens.split([v_input * n_patches, n_patches], dim=1)
        conf_log_var = self.confidence_head(mid_target_tokens) 

        height, width = self.image_size_H, self.image_size_W

        conf_log_var = rearrange(conf_log_var, '(b v) (h w) 1 -> b v h w', 
                                 b=b, v=v_target, h=height // self.patch_size, w=width // self.patch_size)
        
        return -conf_log_var

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]

        checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        
        self.load_state_dict(checkpoint["model"], strict=False)
        print(f"Scene Decoder Analysis Load {ckpt_paths[-1]}")
        return 0
