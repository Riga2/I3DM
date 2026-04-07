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
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

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
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        bias=False,
        dropout=0.0,
        activation=nn.GELU,
        mlp_dim=None,
    ):
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
        
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_bias=None, return_weights=False, split_query_start_idx=None):
        """
        Args:
            x: [b, seq_len, dim]
            return_weights: If True, return attention weights.
            split_query_start_idx: If provided, only calculate attention weights for queries starting from this index.
                                   Useful to reduce memory usage when we only care about Target -> Input attention.
        """
        B, L, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # [b, l, nh, dh]
        q = rearrange(q, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        k = rearrange(k, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        v = rearrange(v, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_weights = None

        if return_weights:
            # Manual Attention calculation for analysis
            # Transpose to [b, nh, l, dh]
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            
            # Determine Query range for attention map
            if split_query_start_idx is not None:
                # Optimized: Only compute attention map for the queries we care about (Target frames)
                # This reduces the Attention Map size from [B, H, ALL, ALL] to [B, H, Target_Len, ALL]
                q_t = q_t[:, :, split_query_start_idx:, :]
                # k_t MUST remain full to ensure Softmax considers all keys (Inputs + Targets)
                # If we slice k_t, we are changing the attention distribution logic.

            scale = 1.0 / math.sqrt(self.head_dim)
            
            # [b, nh, L_q, L_k]
            # This matmul can still be large [B, H, Tgt, All]. 
            # If OOM occurs here, we would need to chunk Q processing, but usually standard attention fits for inference.
            # Softmax over the last dimension (all keys)
            attn_weights = torch.softmax(torch.matmul(q_t, k_t.transpose(-2, -1)) * scale, dim=-1)
            
        # Use efficient attention for the actual output calculation to maintain consistency/speed
        # Note: In strict analysis mode, small numerical differences between xops and manual calc might exist
        if xops is not None:
            x = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            # Fallback manual calculation if xops not available
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.attn_dropout if self.training else 0.0)
            x = x_out.transpose(1, 2)

        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        return x, attn_weights

class QK_Norm_TransformerBlock(nn.Module):
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


    def forward(self, x, return_weights=False, split_query_start_idx=None):
        attn_out, weights = self.attn(self.norm1(x), return_weights=return_weights, split_query_start_idx=split_query_start_idx)
        # In-place add used
        x.add_(attn_out)
        x.add_(self.mlp(self.norm2(x)))
        return x, weights

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

class SceneDecoderOnlyAnalysis(nn.Module):
    def __init__(self, load_path=None):
        super().__init__()
        self.process_data = ProcessDataPlucker()

        self.input_pose_in_channels = 9
        self.target_pose_in_channels = 6
        self.patch_size = 8
        self.transformer_dim = 768
        self.transformer_d_head = 64
        self.n_layer = 24
        self.grad_checkpoint_every = 1

        self.image_size_H = 352
        self.image_size_W = 640

        self.use_qk_norm = True
        self.special_init = True
        self.depth_init = True

        self._init_tokenizers()
        self._init_transformer()

        if load_path is not None:
            self.load_ckpt(load_path)

    def _create_tokenizer(self, in_channels, patch_size, d_model):
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
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.input_pose_in_channels,
            patch_size = self.patch_size,
            d_model = self.transformer_dim
        )
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.target_pose_in_channels,
            patch_size = self.patch_size,
            d_model = self.transformer_dim
        )
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
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                self.transformer_dim, self.transformer_d_head, use_qk_norm=self.use_qk_norm
            ) for _ in range(self.n_layer)
        ]
        
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

    def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1, 
                   return_dependency=False, dependency_split_idx=None):
        num_layers = len(self.transformer_blocks)
        accumulated_dependency = None # [Batch, Input_Len]

        if return_dependency:
            # if gradient_checkpoint:
            #     print("Warning: Gradient checkpointing disabled for dependency analysis.")
                
            for layer in self.transformer_blocks:
                input_tokens, weights = layer(input_tokens, return_weights=True, split_query_start_idx=dependency_split_idx)
                
                # Immediate dependency reduction to save memory
                # weights shape: [B, NH, Tgt_Len, All_Len]
                
                if dependency_split_idx is not None:
                    # Only keep attention to Input section
                    # view operation, minimal cost
                    w_input = weights[..., :dependency_split_idx]
                else:
                    w_input = weights
                
                # Sum over Heads (1) and Target Pixels (2) to get total emphasis on each Input Pixel
                # [B, NH, Tgt, Input] -> [B, Input]
                layer_score = w_input.sum(dim=(1, 2))
                
                # Accumulate immediately
                if accumulated_dependency is None:
                    accumulated_dependency = layer_score
                else:
                    accumulated_dependency.add_(layer_score)
                
                # Explicitly delete to encourage freeing memory
                del weights, w_input, layer_score
                
            return input_tokens, accumulated_dependency
            
        # Normal inference path with checkpointing
        def _process_layer_group(tokens, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                tokens, _ = self.transformer_blocks[idx](tokens) 
            return tokens

        if not gradient_checkpoint:
            for layer in self.transformer_blocks:
                input_tokens, _ = layer(input_tokens)
            return input_tokens, None

        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens, None

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
        else:
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)

    def forward(self, input_data_batch, target_data_batch, ret_imgs=False, return_dependency=False, target_chunk_size=1):
        """
        Args:
            return_dependency: If True, returns a 'dependency_score' indicating how much target pixel depends on each input frame.
            target_chunk_size: Process target frames in chunks to reduce memory usage. Default is 1.
        """
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
        
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b * v_target, n_patches, d]

        combined_bs = b * v_target
        
        all_rendered_feats = []
        accumulated_dependency_list = []
        
        # Determine Input Length
        input_len = input_img_tokens.shape[1]
        
        for start_idx in range(0, combined_bs, target_chunk_size):
            end_idx = min(start_idx + target_chunk_size, combined_bs)
            
            # 1. Slice Target Tokens
            # [chunk_bs, n_patches, d]
            chunk_target_tokens = target_pose_tokens[start_idx:end_idx]
            current_chunk_bs = chunk_target_tokens.shape[0]

            # 2. Prepare Input Tokens for this chunk
            # Each target in flattened batch corresponds to a specific video in 'b'.
            # Map flattened index to batch index: idx // v_target
            chunk_indices = torch.arange(start_idx, end_idx, device=input_img_tokens.device)
            batch_indices = chunk_indices // v_target
            
            # [chunk_bs, input_len, d]
            chunk_input_tokens = input_img_tokens[batch_indices]

            # 3. Concatenate
            transformer_input = torch.cat((chunk_input_tokens, chunk_target_tokens), dim=1)  
            concat_img_tokens = self.transformer_input_layernorm(transformer_input)
            
            # 4. Pass Layers
            transformer_output_tokens, chunk_dependency = self.pass_layers(
                concat_img_tokens, 
                gradient_checkpoint=True, 
                checkpoint_every=self.grad_checkpoint_every,
                return_dependency=return_dependency,
                dependency_split_idx=input_len # Only calculate attention from Target Tokens (Queries) -> All Tokens (Keys)
            )

            # 5. Extract Output
            _, target_image_tokens = transformer_output_tokens.split(
                [input_len, current_chunk_bs * n_patches], dim=1
            ) # Note: split size for target is calculated based on current chunk

            # Decode to features [chunk_bs, n_patches_tgt, d_out]
            # out_feats = self.image_token_decoder(target_image_tokens) # wait, decoder output is [chunk_bs * n_patches, 3 * patch^2]? 
            # No, decoder is map per token.
            # target_image_tokens shape: [chunk_bs, n_patches, d]
            out_feats = self.image_token_decoder(target_image_tokens)
            
            all_rendered_feats.append(out_feats)
            
            if return_dependency and chunk_dependency is not None:
                accumulated_dependency_list.append(chunk_dependency)
                
            # Clear intermediates
            del transformer_input, concat_img_tokens, transformer_output_tokens, chunk_target_tokens, chunk_input_tokens
        
        # Concatenate results
        # [combined_bs, n_patches, (patch_size**2) * 3]
        out_feats = torch.cat(all_rendered_feats, dim=0)

        height, width = self.image_size_H, self.image_size_W

        rendered_images = rearrange(
            out_feats, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // self.patch_size, 
            w=width // self.patch_size, 
            p1=self.patch_size, 
            p2=self.patch_size, 
            c=3
        )

        result = {
            "rendered_images": rendered_images
        }
        
        if return_dependency and len(accumulated_dependency_list) > 0:
            # concatenated dependency: [combined_bs, Input_Len]
            accumulated_dependency = torch.cat(accumulated_dependency_list, dim=0)
            
            # accumulated_dependency: [BS, Input_Len]
            # Reshape Input_Len to [V_input, N_patches]
            # BS is combined_bs
            total_attn = accumulated_dependency.view(combined_bs, v_input, n_patches)
            
            # Sum over spatial patches to get per-frame dependency
            # [Combined_BS, V_input]
            frame_dependency = total_attn.sum(dim=2)
            
            # --- Reliability Score Calculation ---
            # Measure how much attention is focused on Inputs vs Self/Targets
            # Raw sum over all input frames
            total_mass_on_inputs = frame_dependency.sum(dim=-1) # [Combined_BS]
            
            # Normalize by theoretical max attention mass (Layers * Heads * TargetParams)
            # Note: accumulated_dependency sums over Heads and Target Pixels in pass_layers
            num_heads = self.transformer_dim // self.transformer_d_head
            
            # Denominator: Max possible attention sum = N_Layers * N_Heads * N_Target_Pixels * 1.0
            # N_Target_Pixels is same as n_patches here (assuming same resolution)
            denom = self.n_layer * num_heads * n_patches
            
            reliability_score = total_mass_on_inputs / (denom + 1e-6)
            result["reliability_score"] = reliability_score.view(b, v_target)
            # -------------------------------------

            # Normalize to get relative distribution (sum=1 across input frames)
            # Use in-place operations
            sum_val = frame_dependency.sum(dim=-1, keepdim=True)
            sum_val.add_(1e-6)
            frame_dependency.div_(sum_val)
            
            # Reshape back to [b, v_target, v_input]
            result["dependency_map"] = frame_dependency.view(b, v_target, v_input)

        return result

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
