## DiT with MoE \ Mamba Block

This repo is changed from [fast DiT](https://github.com/chuanyangjin/fast-DiT). Provide two different blocks. <br> One is based on MoE. Another is re-implemented the bi-mamba block, refer to [DIFFUSSM](https://arxiv.org/ads/2311.18257) and [Vim](https://github.com/hustvl/Vim) 

see [`models.py`](models.py) for details.

### Preparation Before Training
To extract ImageNet features with `1` GPUs on one node:

```bash
bash extract_feature.sh
```

### Training
To launch DiT-XL/2 (256x256) training with `N` GPUs on one node:
```bash
bash train.sh
```

## Evaluation (FID, Inception Score, etc.)

```bash
bash sample_ddp.sh
```

generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and other metrics. 

## Reference

