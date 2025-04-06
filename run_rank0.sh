#!/bin/bash

# --- Configuration ---
MASTER_TAILSCALE_IP="100.95.237.88"
TAILSCALE_IFNAME="tailscale0"
PORT="9033"
NUM_LAYERS_PER_STAGE=24 # 48 / 2 stages

# --- Base Arguments ---
# Changed --load-pretrained-model to false
# Changed --model-name to gpt2-xl (or keep checkpoints/gpt2-xl)
ARGS="--model-name gpt2-xl \
--tokenizer-name gpt2-xl \
--load-pretrained-model false \
--task-name arxiv21 --n-epochs 10 --warmup-epochs 1 \
--num-heads 25 --embedding-dim 1600 \
--num-iters 10000000 --lr 5e-5 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
--forward-compress-method delta \
--forward-bits 4 \
--backward-compress-method fixpoint \
--backward-bits 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

# --- Environment Variables ---
export GLOO_SOCKET_IFNAME=${TAILSCALE_IFNAME}
export NCCL_SOCKET_IFNAME=${TAILSCALE_IFNAME}
export CUDA_VISIBLE_DEVICES=0

# --- Launch Command ---
echo "Starting Rank 0 (Training from Scratch)..."
python dist_lm_runner.py \
    $(echo ${ARGS}) \
    --num-layers ${NUM_LAYERS_PER_STAGE} \
    --dist-url tcp://${MASTER_TAILSCALE_IP}:${PORT} \
    --world-size 2 \
    --pipeline-group-size 2 \
    --data-group-size 1 \
    --rank 0 \
    --cuda-id 0

echo "Rank 0 finished."