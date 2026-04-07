

import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils 
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer
import torch.nn.functional as F


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


class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessDataLatest(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize loss computer
        # self.loss_computer = LossComputer(config)
        
        self.exit_layer = 6  
        print(f"Initialized model with exit_layer={self.exit_layer} for confidence prediction.")

        if self.config.training.load_pretrained:
            ckpt_path = self.config.training.pretrained_ckpt_path
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except:
                traceback.print_exc()
                print(f"Failed to load {ckpt_path}")
                return None
            self.load_state_dict(checkpoint["model"], strict=False)
            print(f"Loaded pretrained weights from {ckpt_path}")
            
        # Freeze original network
        for p in self.parameters():
            p.requires_grad = False
            
        # Initialize Confidence Head (trainable)
        self.confidence_head = ConvConfidenceHead(
            in_dim=self.config.model.transformer.d
        )
        # Ensure confidence head is trainable
        for p in self.confidence_head.parameters():
            p.requires_grad = True

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
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)

        # Create transformer blocks
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.n_layer)
        ]
        
        # Apply special initialization if configured
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_blocks):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)
                
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(config.d, bias=False)


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        # self.loss_computer.eval()

            
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

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

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
    
    def forward(self, data_batch, has_target_image=True):
        input, target = self.process_data(data_batch, has_target_image=has_target_image, compute_rays=True)

        posed_input_images = self.get_posed_input(images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        b, v_input, c, h, w = posed_input_images.size()
        input_img_tokens = self.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)
        
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)

        repeated_input_img_tokens = repeat(input_img_tokens, 'b np d -> (b v_target) np d', v_target=v_target)
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        
        checkpoint_every = self.config.training.grad_checkpoint_every

        
        mid_tokens = self.pass_layers(
            concat_img_tokens, 
            start_layer=0, 
            end_layer=self.exit_layer, 
            gradient_checkpoint=True, 
            checkpoint_every=checkpoint_every
        )
        
        _, mid_target_tokens = mid_tokens.split([v_input * n_patches, n_patches], dim=1)
        conf_log_var = self.confidence_head(mid_target_tokens) 
        
        final_tokens = self.pass_layers(
            mid_tokens, 
            start_layer=self.exit_layer,
            gradient_checkpoint=True, 
            checkpoint_every=checkpoint_every
        )
        
        # ================================================

        _, final_target_tokens = final_tokens.split([v_input * n_patches, n_patches], dim=1)
        rendered_images = self.image_token_decoder(final_target_tokens)
        
        height, width = target.image_h_w
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target, h=height // patch_size, w=width // patch_size, p1=patch_size, p2=patch_size, c=3
        )
        if has_target_image:
            # --- Confidence Loss Calculation ---
            # We skip the heavy loss_computer (LPIPS, VGG) since the backbone is frozen.
            # We only calculate MSE directly here as it is required for Aleatoric Loss.
            
            # 1. Prepare data: [b, v, c, h, w] -> [b*v, c, h, w]
            flat_rendered = rearrange(rendered_images, 'b v c h w -> (b v) c h w')
            flat_target = rearrange(target.image, 'b v c h w -> (b v) c h w')
            
            # flat_rendered: [b*v, c, h, w]
            # flat_target: [b*v, c, h, w]
            
            mse_pixel = (flat_rendered - flat_target) ** 2 # [b*v, c, h, w]
            mse_pixel = mse_pixel.mean(dim=1)
            
            p = self.config.model.target_pose_tokenizer.patch_size
            mse_patches = F.avg_pool2d(mse_pixel, kernel_size=p, stride=p) # [b*v, h_patches, w_patches]
            mse_patches = rearrange(mse_patches, 'b h w -> b (h w) 1') # [b*v, n_patches, 1]
            
            mse_gt = mse_patches.detach()
            
            uncertainty_loss = 0.5 * torch.exp(-conf_log_var) * mse_gt + 0.5 * conf_log_var
            uncertainty_loss = uncertainty_loss.mean()
            
            # 4. Compute PSNR for monitoring (Cheap)
            with torch.no_grad():
                avg_mse = mse_gt.mean()
                psnr = -10.0 * torch.log10(avg_mse + 1e-8)
            
            # 5. Construct Metrics
            loss_metrics = edict(
                loss=uncertainty_loss, # Only train the head
                
                conf_loss=uncertainty_loss,
                norm_conf_loss=uncertainty_loss / (avg_mse + 1e-8),
                
                # Monitoring metrics
                psnr=psnr,
                l2_loss=avg_mse,
                
                # Placeholders for compatibility (skip expensive computation)
                lpips_loss=torch.tensor(0.0, device=flat_rendered.device),
                perceptual_loss=torch.tensor(0.0, device=flat_rendered.device),
                norm_perceptual_loss=torch.tensor(0.0, device=flat_rendered.device), 
                norm_lpips_loss=torch.tensor(0.0, device=flat_rendered.device)
            )
            
        else:
            loss_metrics = None

        conf_log_var = rearrange(conf_log_var, '(b v) (h w) 1 -> (b v) 1 h w', 
                                 b=b, v=v_target, h=height // patch_size, w=width // patch_size)
        conf_log_var_vis = F.interpolate(conf_log_var, size=(height, width), mode='bilinear', align_corners=False)
        # conf_log_var_vis = rearrange(conf_log_var_vis, '(b v) 1 h w -> b v 1 h w', b=b, v=v_target)
        
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            confidence_score=-conf_log_var_vis, # Quality score (higher is better)
            conf=None,
            gt_conf=None,
            )
        
        return result
    
    def forward_old(self, data_batch, has_target_image=True):

        input, target = self.process_data(data_batch, has_target_image=has_target_image, compute_rays=True)

        # Process input images
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()

        input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        
     
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)

        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v, n_patches, d]

        # Repeat input tokens for each target view
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target, np=n_patches * v_input
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(concat_img_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)

        # Predict confidence (log variance)
        # transformer_output_tokens: [b * v_target, (v+1)*n_patches, d]
        # conf_log_var = self.confidence_head(transformer_output_tokens) # [b * v_target, 1]
        
        # Discard the input tokens
        _, target_image_tokens = transformer_output_tokens.split(
            [v_input * n_patches, n_patches], dim=1
        ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]

        # target_image_tokens: [b * v_target, n_patches, d]
        conf_log_var = self.confidence_head(target_image_tokens)

        # [b*v_target, n_patches, p*p*3]
        rendered_images = self.image_token_decoder(target_image_tokens)
        
        height, width = target.image_h_w

        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )
        if has_target_image:
            # --- Confidence Loss Calculation ---
            # We skip the heavy loss_computer (LPIPS, VGG) since the backbone is frozen.
            # We only calculate MSE directly here as it is required for Aleatoric Loss.
            
            # 1. Prepare data: [b, v, c, h, w] -> [b*v, c, h, w]
            flat_rendered = rearrange(rendered_images, 'b v c h w -> (b v) c h w')
            flat_target = rearrange(target.image, 'b v c h w -> (b v) c h w')
            
            # flat_rendered: [b*v, c, h, w]
            # flat_target: [b*v, c, h, w]
            
            mse_pixel = (flat_rendered - flat_target) ** 2 # [b*v, c, h, w]
            mse_pixel = mse_pixel.mean(dim=1)
            
            p = self.config.model.target_pose_tokenizer.patch_size
            mse_patches = F.avg_pool2d(mse_pixel, kernel_size=p, stride=p) # [b*v, h_patches, w_patches]
            mse_patches = rearrange(mse_patches, 'b h w -> b (h w) 1') # [b*v, n_patches, 1]
            
            mse_gt = mse_patches.detach()
            
            uncertainty_loss = 0.5 * torch.exp(-conf_log_var) * mse_gt + 0.5 * conf_log_var
            uncertainty_loss = uncertainty_loss.mean()
            
            # 4. Compute PSNR for monitoring (Cheap)
            with torch.no_grad():
                avg_mse = mse_gt.mean()
                psnr = -10.0 * torch.log10(avg_mse + 1e-8)
            
            # 5. Construct Metrics
            loss_metrics = edict(
                loss=uncertainty_loss, # Only train the head
                
                conf_loss=uncertainty_loss,
                norm_conf_loss=uncertainty_loss / (avg_mse + 1e-8),
                
                # Monitoring metrics
                psnr=psnr,
                l2_loss=avg_mse,
                
                # Placeholders for compatibility (skip expensive computation)
                lpips_loss=torch.tensor(0.0, device=flat_rendered.device),
                perceptual_loss=torch.tensor(0.0, device=flat_rendered.device),
                norm_perceptual_loss=torch.tensor(0.0, device=flat_rendered.device), 
                norm_lpips_loss=torch.tensor(0.0, device=flat_rendered.device)
            )
            
        else:
            loss_metrics = None

        conf_log_var = rearrange(conf_log_var, '(b v) (h w) 1 -> (b v) 1 h w', 
                                 b=b, v=v_target, h=height // patch_size, w=width // patch_size)
        conf_log_var_vis = F.interpolate(conf_log_var, size=(height, width), mode='bilinear', align_corners=False)
        # conf_log_var_vis = rearrange(conf_log_var_vis, '(b v) 1 h w -> b v 1 h w', b=b, v=v_target)
        
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            confidence_score=-conf_log_var_vis, # Quality score (higher is better)
            conf=None,
            gt_conf=None,
            )
        
        return result

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=False)
        return 0


