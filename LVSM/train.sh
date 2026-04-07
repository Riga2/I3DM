# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_decoder_only_mask_update.yaml

# torchrun --nproc_per_node 8 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask.yaml

# torchrun --nproc_per_node 8 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask_precomputed.yaml

# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask_precomputed.yaml

# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask_tokens_new.yaml

# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask_precomputed_small.yaml

# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask_tokens.yaml

# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask_tokens_new.yaml


# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_encoder_decoder_mask_precomputed_vae.yaml

# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_decoder_only_F8_DL3DV.yaml

# torchrun --nproc_per_node 4 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_decoder_only_occ.yaml

# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 3 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
#     train.py --config configs/LVSM_scene_decoder_only_occ_highRes.yaml

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    train.py --config configs/LVSM_scene_decoder_only_occ.yaml