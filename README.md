# Activation Compression with Guarantees

We explore how to fine-tune language models over slow networks using activation compression with guarantees. 
This is a research project developed by [DS3Lab@ETH Zurich](https://ds3lab.inf.ethz.ch/) and [HazyResearch@Stanford](https://hazyresearch.stanford.edu/).

## Cite Our Paper

```bibtex
@article{jue2022fine,
  title={Fine-tuning Language Models over Slow Networks using Activation Compression with Guarantees}, 
  author={Jue Wang, Binhang Yuan, Luka Rimanic, Yongjun He, Tri Dao, Beidi Chen, Christopher Re, Ce Zhang},
  year={2022},
  eprint={2206.01299},
  archivePrefix={arXiv},
  primaryClass={cs.DC}
}
```

## AWS AMI

You can directly use our AWS AMI for easy configuration: 

| AMI Name       | AMI ID                | Region    | Recommended Instances                |
|----------------|-----------------------|-----------|--------------------------------------|
| 	ac-sgd-jun-9  | ami-02870383e79fd0a48 | us-west-2 | p3.2xlarge, p3.8xlarge, p3.16xlarge  |

## Setup:

### Use Our AWS AMI (Recommended)

- Launch instances from [our AMI](https://us-west-2.console.aws.amazon.com/ec2/v2/home?region=us-west-2#ImageDetails:imageId=ami-02870383e79fd0a48).

- Activate Pytorch env:
  ```bash
  source activate pytorch_p38
  ```

- Update code (Optional):
  ```bash
  cd AC-SGD
  git pull
  ```
  
- Setup network configuration:

  ```bash
  export GLOO_SOCKET_IFNAME=ens3
  export NCCL_SOCKET_IFNAME=ens3
  ```

### Setup Manually

- Create environment:

  ```bash
  conda create -n acsgd python=3.8
  conda activate acsgd
  ```

- Install PyTorch env: 

  ```bash
  pip3 install torch==1.9.0+cu111 torchtext -f https://download.pytorch.org/whl/torch_stable.html

  # Magic, not sure why cupy-cuda111 would not work, it seems that cupy-cuda111 will use different PTX from torch.
  pip3 install cupy-cuda110==8.6.0
  ```
  
  Other dependencies:
 
  ```bash
  pip3 install datasets==2.2.2
  pip3 install transformers==4.19.2
  pip3 install sentencepiece==0.1.96 # required by deberta
  ```
  
- Download datasets:

  ```bash
  wget https://gpt-activation-compression.s3.us-east-2.amazonaws.com/data.zip
  unzip data.zip
  ```
  
- Setup network configuration:

  ```bash
  export GLOO_SOCKET_IFNAME=ens3
  export NCCL_SOCKET_IFNAME=ens3
  ```

## Run Distributed Gpipe:

- Partition the pre-trained model:
  
  ```bash
  # gpt2
  python convert_gpt2_checkpoint --model-name gpt2-xl --save-dir checkpoints/
      
  # or deberta 
  python convert_deberta_checkpoint --model-name deberta-v2-xxl --save-dir checkpoints/
  ```

- On each node, run:
  
  ```bash
  # gpt2
  python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank i # (i=0,...,N-1)
      
  # or deberta
  python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank i # (i=0,...,N-1)
  ```
  where "ARGS" contains training-related configurations, which should remain the same across nodes. An example could be:
  ```bash
  ARGS="--model-name checkpoints/gpt2-xl \
    --tokenizer-name gpt2-xl \
    --load-pretrained-model true \
    --task-name wikitext --n-epochs 10 --warmup-epochs 1 \
    --num-layers 6 --num-heads 25 --embedding-dim 1600 \
    --num-iters 10000000 --lr 5e-5 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
    --forward-compress-method delta \
    --forward-bits 4 \
    --backward-compress-method fixpoint \
    --backward-bits 8 \
    --dist-url tcp://XXX.XXX.XXX.XXX:9000 \
    --world-size N --pipeline-group-size N \
    --pp-mode gpipe --profiling no-profiling --do-evaluation true"
  ```
  Modify `"--dist-url"`, `"--world-size"` and `"--pipeline-group-size"` before running.
  
  Complete examples can be found "./run_lm.sh" and "./run_deberta.sh".
  
  
## Arguments

### Distributed Related

- `"--dist-url"`: tcp://XXX.XXX.XXX.XXX:9000
- `"--world-size"`: number of nodes that participate in the training.
- `"--pipeline-group-size"`: number of nodes that perform pipeline parallelism.
- `"--data-group-size"`: number of nodes that perform data parallelism.
- `"--rank"`: the rank of the current node. (0, ..., world_size-1)
- `"--profiling"`: "no-profiling" or "tidy_profiling". If "tidy_profiling", a trace file will be generated in "./trace_json/", which can be visualized with "chrome://tracing/".

### Compression Related

- `"--forward-compress-method"`: "none", "fixpoint", "delta", or "delta-lowbits".
  - "none": do not compress.
  - "fixpoint": direct compress the activations. need to specify `"--forward-bits".
  - "delta": compress and communicate the delta of activations. need to specify `"--forward-bits"` and `"--max-activation-cache-size"`.
  - "delta-lowbits": in addition to "delta", it also compresses the local cache (previous activations). need to specify `"--forward-bits"`, `"--forward-bits-act"`, and `"--max-activation-cache-size"`.
- `"--backward-compress-method"`: "none" or "fixpoint".
  - "none": do not compress.
  - "fixpoint": direct compress the gradients. need to specify `"--backward-bits"`.

### Training Related

- `"--batch-size"`: macro-batch size.
- `"--micro-batch-size "`: micro-batch-size. The macro-batch size should be divisible by micro-batch-size.
- `"--lr"`: the peak learning rate.
- `"--n-epochs"`: number of training epochs.
- `"--warmup-epochs"`: number of epochs for uncompressed training (transfer full-precision activations and gradients).
- `"--warmup-steps"`: number of training steps where the learning rate grows from 0 to `"--lr"`. Default to be one training epoch.
- `"--do-evaluation"`: whether do evaluation during training.

### Model Related

- `"--model-name"`: Name or path of the pretrained checkpoint. Usually should be a path to the checkpoint generated by "convert_xxx_checkpoint.py".
- `"--tokenizer-name"`: Name or path of the tokenizer.
- `"--load-pretrained-model"`: whether to load the pretrained checkpoint. The checkpoint should be generated by "convert_xxx_checkpoint.py".
- `"--num-layers", `"--num-heads", `"--embedding-dim"` should be inline with the configuration of `"--model-name"`.
