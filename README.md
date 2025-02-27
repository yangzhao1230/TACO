# Regulatory DNA Sequence Design with Reinforcement Learning

This repository provides the official implementation for the ICLR 2025 poster paper:  
["Regulatory DNA Sequence Design with Reinforcement Learning"](https://openreview.net/forum?id=F4IMiNhim1).

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
Recently, we found that different versions of FlashAttention can lead to variations in results, possibly related to the implementation of HyenaDNA. Specifically, even with the same model weights, inference may produce slightly different intermediate outputs, which could result in a 1-2 point variation in the final evaluation metric. Unfortunately, we did not record the exact version of FlashAttention during the ICLR submission period. We are currently working to reproduce and diagnose this issue.

To install FlashAttention, run:

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Dao-AILab/flash-attention.git

# From inside flash-attn/
cd flash-attention
cd csrc/layer_norm && pip install .
```

---

## TFFBS Inference
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
Our implementation is based on **regLM, LatProtRL, and ChemformerRL**.  
We sincerely appreciate their contributions!

---

## Citation
TBD
```
