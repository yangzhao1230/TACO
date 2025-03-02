# Regulatory DNA Sequence Design with Reinforcement Learning

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

**Note:** We recently observed that variations in FlashAttention versions may cause slight differences in results, potentially due to interactions with HyenaDNA. Specifically, even using identical model weights, inference might yield slightly different intermediate outputs, leading to about 1–2 points variation in the final evaluation metric. Unfortunately, we did not record the exact FlashAttention version during the ICLR submission period (and the development machine from that period has since been recycled). We are actively working to reproduce and investigate this issue.


---

## Data Preparation

Our data preprocessing scripts are mainly adapted from **[regLM](https://github.com/Genentech/regLM)**, with additional processing steps for extracting TFBS features. Here, we provide scripts for TFBS feature extraction along with our processed data. You can integrate your own data splits and customize the pipeline based on the original regLM scripts.

TBD

## TFBS Reward Inference

TBD

---

## Optimization with RL

We provide only the inference script for **offline MBO (Section 4.3)** in the paper.  
However, the implementations of **Section 4.2 and Section 4.3** are **identical**, except for differences in the **reward model, pre-trained model, and dataset**.

To run inference for offline MBO, use:

```bash
bash reinforce_mbo.sh
```

---

## Acknowledgements
Our implementation is based on **[regLM](https://github.com/Genentech/regLM)**, **[LatProtRL](https://github.com/haewonc/LatProtRL)**, and **[RL4Chem](https://github.com/montrealrobotics/RL4Chem)**.  
We sincerely appreciate their valuable contributions.

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
