# 🌮 TACO: Regulatory DNA Sequence Design with Reinforcement Learning

This repository provides the official implementation for the ICLR 2025 poster paper:  
["Regulatory DNA Sequence Design with Reinforcement Learning"](https://openreview.net/pdf?id=F4IMiNhim1).

## Environment Setup

Before running the code, create and activate a Conda environment:

```bash
conda create -n taco python=3.9
conda activate taco
```

To install all necessary dependencies, run:

```bash
bash env_install.sh
```


To install FlashAttention, run:

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Dao-AILab/flash-attention.git

# From inside flash-attn/
cd flash-attention
cd csrc/layer_norm && pip install .
```

**Note:** We have recently observed that variations in FlashAttention versions may lead to slight differences in results, potentially due to interactions with HyenaDNA. Specifically, even when using identical model weights, inference can produce slightly different intermediate outputs (for example, we found that a sample had identical output bases until position 131, but diverged from position 132 onward compared to our previous experimental results), resulting in approximately 1-2 point variations in the final evaluation metrics. Unfortunately, we did not document the exact FlashAttention version used during the ICLR submission period (and the development machine from that time has since been recycled). We are actively working to reproduce and investigate this issue. The table below presents our reproduction results from February 2025 as shown in `calculate_metric.ipynb`: 


|  |                      |  |  |
|-----------------|----------------------|----------------------|----------------------|
| SK-N-SH Results                | Top                  | Medium               | Diversity            |
| alpha = 0.0 (Paper reported)  | 0.67 ± 0.06          | 0.60 ± 0.06          | 111.6 ± 12.86        |
| alpha = 0.01 (Paper reported)    | 0.68 ± 0.08          | 0.62 ± 0.08          | 121.4 ± 7.86         |
| alpha = 0.0 (Latest reproduction) | 0.68 ± 0.07      | 0.62 ± 0.07          | 120.2 ± 13.85        |
| alpha = 0.01 (Latest reproduction)     | 0.7 ± 0.03           | 0.63 ± 0.04          | 117.2 ± 12.64        |

---

## Data Preparation and Reward Model Training

Our data preprocessing scripts and reward model training scripts are mainly adapted from regLM (https://zenodo.org/records/12668907). Specifically, the code repository structure of regLM is as follows:

```
- human_enhancers
 - 01_data_processing
 - 02_regression_paired
 - 03_regression_separate
 - 04_reglm
 - 05_reglm_interpretation
 - 06_synthetic_enhancer_generation
 - 07_synthetic_enhancer_evaluation
 - 08_synthetic_enhancer_comparison
```
You can find instructions for downloading datasets (and splitting them according to your specific needs) in `01_data_processing`, and learn how to train reward models in `02_regression_paired`.

In particular, if you want to enable TFBS features, you need to first scan and extract TFBS features from sequence data (also in `01_data_processing`). Briefly, you should save the TFBS feature matrix of interest and the corresponding fitness levels into an `h5ad` format file. Regarding the related JASPAR files needed for scanning TFBS, please refer to https://github.com/yangzhao1230/TACO/issues/2.

## TFBS Reward Inference

You should have extracted TFBS features and saved them in `h5ad` format (please refer to regLM's notebooks for this step).

Then you can train the lightGBM model with 
```
python lightGBM_mbo.py
```
then you can infer the TFBS reward through `tfbs_reward_mbo.ipynb`.

## Prepared Data, TFBS Reward and Reward Model

Here we provide pre-processed TFBS rewards, surrogate scoring model weights (note that the policy directly uses regLM weights, and for oracle weights please use the reward model provided by regLM), and datasets partitioned according to the offline MBO strategy described in the paper.

**Model weights:** https://huggingface.co/yangyz1230/TACO/tree/main  
**Datasets:** https://huggingface.co/datasets/yangyz1230/TACO/tree/main

## Optimization with RL

We provide only the inference script for **offline MBO (Section 4.3)** in the paper.  
However, the implementations of **Section 4.2 and Section 4.3** are **identical**, except for differences in the **reward model, pre-trained model, and dataset**.

To run inference for offline MBO, use:

```bash
bash reinforce_mbo.sh
```

---

## Acknowledgements

Our implementation builds upon several open-source projects:
- **[regLM](https://github.com/Genentech/regLM)**: Provided the implementation of our policy, reward model, and data processing related code.
- **[LatProtRL](https://github.com/haewonc/LatProtRL)**: Contributed baseline implementations and evaluation code
- **[RL4Chem](https://github.com/montrealrobotics/RL4Chem)**: Supplied the reinforcement learning algorithmic framework

We sincerely appreciate their valuable contributions to this work.

---

## TODO List
- [x] Provide environment configuration instructions.
- [x] Provide core algorithm code implementation.
- [x] Provide data and checkpoints for offline MBO settings.
- [x] Provide the training scripts for the LightGBM model.
- [x] Provide code for tfbs reward inference.

---

## Citation
If you use our code or find our work inspiring, please cite our paper:

```bibtex
@inproceedings{yang2025regulatory,
  title={Regulatory DNA Sequence Design with Reinforcement Learning},
  author={Zhao Yang and Bing Su and Chuan Cao and Ji-Rong Wen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=F4IMiNhim1}
}
