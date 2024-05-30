# Mask Grounding
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%20|%202.3.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)


This repository contains official code for **CVPR2024** paper:
> [Mask Grounding for Referring Image Segmentation](https://arxiv.org/abs/2312.12198)  


## Introduction
Mask Grounding is an innovative auxiliary task that can significantly improve the performance of existing Referring Image Segmentation algorithms by explicitly teaching these models to learn fine-grained visual grounding in their language features. Specifically, during training, these models are given sentences with randomly masked textual tokens and must learn to predict the identities of these tokens based on their surrounding textual, visual and segmentation information.

<div align="center">
  <img src="https://github.com/yxchng/mask-grounding/blob/main/imgs/fig1.jpg?raw=true" width="100%" height="100%"/>
</div><br/>

## Results

| Backbone | RefCOCO (val) | RefCOCO (testA) | RefCOCO (testB) | RefCOCO+ (val) | RefCOCO+ (testA) | RefCOCO+ (testB) | G-Ref (val(U)) | G-Ref (test(U)) | G-Ref (val(G)) |
|---|---|---|---|---|---|---|---|---|---|
| CRIS | 70.47 | 73.18 | 66.10 | 62.27 | 68.08 | 53.68 | 59.87 | 60.36 | - |
| LAVT | 72.73 | 75.82 | 68.79 | 62.14 | 68.38 | 55.10 | 61.24 | 62.09 | 60.50 |
| ReLA | 73.82 | 76.48 | 70.18 | 66.04 | 71.02 | 57.65 | 65.00 | 65.97 | 62.70 |
| MagNet (Ours) | 75.24 | 78.24 | 71.05 | 66.16 | 71.32 | 58.14 | 65.36 | 66.03 | 63.13 |

## Citation
```
@inproceedings{chng2023mask,
  title={Mask Grounding for Referring Image Segmentation},
  author={Chng, Yong Xien and Zheng, Henry and Han, Yizeng and Qiu, Xuchong and Huang, Gao},
  booktitle={CVPR},
  year={2024}
}
```

## Reference
This code is built on [LAVT](https://github.com/openai/CLIP), [Mask2Former-Simplify](https://github.com/zzubqh/Mask2Former-Simplify), [ovr-cnn](https://github.com/alirezazareian/ovr-cnn).
