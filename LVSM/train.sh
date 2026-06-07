cd "$(dirname "$0")"

torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    train.py --config configs/LVSM_scene_decoder_only_occ.yaml \
    "$@"