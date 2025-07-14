# [ICCV 25] CWNet: Causal Wavelet Network for Low-Light Image Enhancement
✨ Code & paper coming soon.

## Abstract
Traditional Low-Light Image Enhancement (LLIE) methods primarily focus on uniform brightness adjustment, often neglecting instance-level semantic information and the inherent characteristics of different features. To address these limitations, we propose CWNet (Causal Wavelet Network), a novel architecture that leverages wavelet transforms for causal reasoning. Specifically, our approach comprises two key components: 1) Inspired by the concept of intervention in causality, we adopt a causal reasoning perspective to reveal the underlying causal relationships in low-light enhancement. From a global perspective, we employ a metric learning strategy to ensure causal embeddings adhere to causal principles, separating them from non-causal confounding factors while focusing on the invariance of causal factors. At the local level, we introduce an instance-level CLIP semantic loss to precisely maintain causal factor consistency. 2) Based on our causal analysis, we present a wavelet transform-based backbone network that  effectively  optimizes the recovery of frequency information, ensuring precise enhancement tailored to the specific attributes of wavelet transforms. Extensive experiments demonstrate that CWNet significantly outperforms current state-of-the-art methods across multiple datasets, showcasing its robust performance across diverse scenes.

## Installation
```
conda env create -f environment.yml
conda activate cwnet
```
## CWNet


## Train
```
python train.py -opt ./options/train/huawei.yml
```

## Test
```
python test.py -opt ./options/test/huawei.yml
```

## Pre-trained Models and Outputs
✨ [Here](https://drive.google.com/drive/folders/1Bcom7bANqh1_m2rNgEuG7C_JAAAF1bEh?usp=sharing).
