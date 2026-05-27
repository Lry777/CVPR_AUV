# CVPR_AUV
# Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models (CVPR 2026)

[cite_start]Official PyTorch implementation of the paper: **"Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models"**.

---

## 📖 Introduction

Medical image segmentation provides critical support for clinical workflows. However, medical image datasets are often affected by acquisition noise and annotation ambiguity, leading to pervasive data uncertainty (aleatoric uncertainty) that substantially undermines model robustness. While existing studies focus primarily on model architectural improvements and predictive reliability estimation, a systematic exploration of the intrinsic data uncertainty remains insufficient.

To address this gap, this work proposes leveraging the universal representation capabilities of visual foundation models to estimate inherent data uncertainty. Specifically, we analyze the feature diversity of the model's decoded representations and quantify their singular value energy to define the **semantic perception scale** for each class, thereby measuring sample difficulty and aleatoric uncertainty without relying on annotations. 

Based on this foundation, we design two uncertainty-driven application strategies:
1. **Aleatoric Uncertainty-Aware Data Filtering**: To eliminate potentially noisy samples and enhance model learning quality.
2. **Dynamic Uncertainty-Aware Optimization (DUO)**: An adaptive optimization strategy that dynamically adjusts class-specific loss weights during training based on the semantic perception scale, combined with a label denoising mechanism to improve training stability.

---

## 🛠️ Methodology Pipeline

![Methodology Pipeline](path/to/your/pipeline_image.png)
[cite_start]*Figure: Overview of our proposed framework, illustrating the quantification of aleatoric uncertainty and its two major applications (Data Filtering and DUO)[cite: 440, 441].*

---

## 💻 Core Implementation & Examples
