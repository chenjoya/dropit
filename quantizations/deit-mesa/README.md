## Demo project for training DeiT with Mesa

For more details, please checkout our paper: [Mesa: A Memory-saving Training Framework for Transformers](https://arxiv.org/abs/2111.11124).

## Usage

1. Install Mesa from [here](https://github.com/zhuang-group/Mesa).
2. Install timm.
    ```bash
    pip install timm==0.3.2
   ```
3. To train a model.
   ```bash
   conda activate mesa
   bash scripts/run.sh [model] [gpus]

   # For example, to train DeiT-Ti with Mesa on 1 GPU

   bash scripts/run.sh deit_tiny_patch16_224 1
   # You may need to change the `--data-path` and `--data-set` (CIFAR or IMNET) in scripts/run.sh to make sure you have the correct path to dataset.
   ```

## Results on ImageNet


| Model               | Param (M) | FLOPs (G) | Train Memory | Top-1 (%) |
| ------------------- | --------- | --------- | ------------ | --------- |
| DeiT-Ti             | 5         | 1.3       | 4,171         | 71.9      |
| **DeiT-Ti w/ Mesa** | 5         | 1.3       | **1,858**     | **72.1**  |
| DeiT-S              | 22        | 4.6       | 8,459         | 79.8      |
| **DeiT-S w/ Mesa**  | 22        | 4.6       | **3,840**     | **80.0**    |
| DeiT-B              | 86        | 17.5      | 17,691        | 81.8      |
| **DeiT-B w/ Mesa**  | 86        | 17.5      | **8,616**     | **81.8**  |



## Acknowledgments

This repository has adopted codes from [DeiT](https://github.com/facebookresearch/deit), we thank the authors for their open-sourced code.
