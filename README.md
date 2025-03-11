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

**Note:** We have recently observed that variations in FlashAttention versions may lead to slight differences in results, potentially due to interactions with HyenaDNA. Specifically, even when using identical model weights, inference can produce slightly different intermediate outputs (for example, we found that a sample had identical output bases until position 131, but diverged from position 132 onward compared to our previous experimental results), resulting in approximately 1-2 point variations in the final evaluation metrics. Unfortunately, we did not document the exact FlashAttention version used during the ICLR submission period (and the development machine from that time has since been recycled). We are actively working to reproduce and investigate this issue. The table below presents our reproduction results from February 2025 as shown in calculate_metric.ipynb: 


| SK-N-SH Results |                      |  |  |
|-----------------|----------------------|----------------------|----------------------|
|                 | Top                  | Medium               | Diversity            |
| alpha = 0.0 (Paper reported)  | 0.67 ± 0.06          | 0.60 ± 0.06          | 111.6 ± 12.86        |
| alpha = 0.01 (Paper reported)    | 0.68 ± 0.08          | 0.62 ± 0.08          | 121.4 ± 7.86         |
| alpha = 0.0 (Latest reproduction) | 0.68 ± 0.07      | 0.62 ± 0.07          | 120.2 ± 13.85        |
| alpha = 0.01 (Latest reproduction)     | 0.7 ± 0.03           | 0.63 ± 0.04          | 117.2 ± 12.64        |

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

## TODO List
<ul>
   <li><input type="checkbox" checked> Provide environment configuration instructions.</li>
   <li><input type="checkbox" checked> Provide core algorithm code implementation.</li>
   <li><input type="checkbox"> Replace all absolute paths in the repo and provide appropriate path instructions.</li>
   <li><input type="checkbox"> Provide checkpoints for pre-trained policy, surrogate, and oracle.</li>
   <li><input type="checkbox"> Provide the construction diamagnetic for tbfs features.</li>
   <li><input type="checkbox"> Provide code for tfbs reward inference.</li>
</ul>

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
