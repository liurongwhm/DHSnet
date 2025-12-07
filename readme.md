
<div align="center">

<h1>DHSnet: Dual Classification Head Self-training Network for Cross-scene Hyperspectral Image Classification</h1>

<h2>IEEE Transactions on Geoscience and Remote Sensing</h2>


[Rong Liu](https://scholar.google.com/citations?user=pOXE8p8AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Junye Liang](https://scholar.google.com/citations?hl=zh-CN&user=cQAAdBYAAAAJ)<sup>1 </sup>, [Jiaqi Yang](https://github.com/liurongwhm)<sup>2 †</sup>, [Meiqi Hu](https://scholar.google.com/citations?user=E-loHKYAAAAJ&hl=zh-CN&oi=ao)<sup>1 †</sup>, [Peng Zhu](https://scholar.google.com/citations?hl=zh-CN&user=iao5Lp0AAAAJ)<sup>3</sup>, [Liangpei Zhang](https://github.com/liurongwhm)<sup>4</sup>

<sup>1</sup> Sun Yat-sen University, <sup>2</sup> Wuhan University,  <sup>3</sup> James Cook University, <sup>3</sup> The University of Hong Kong, <sup>4</sup> Henan Academy of Sciences.

<sup>†</sup> Corresponding author

</div>


# 🌞 Overview

**Dual Classification Head Self-training Network (DHSnet)** is a novel framework for cross-scene HSI classification. It aligns class-wise features across domains, ensuring that the trained classifier can accurately classify TD data of different classes. We introduce a dual classification head self-training strategy for the first time in the cross-scene HSI classification field and design a self-training loss based on the prediction of the two classification heads. The proposed approach mitigates the domain gap while preventing the accumulation of incorrect pseudo-labels in the model. Additionally, we incorporate a novel central feature attention mechanism to enhance the model’s capacity to learn scene-invariant features across domains. DHSNet significantly outperforms state-of-the-art methods on three cross-scene HSI datasets, achieving 80.23±1.92% OA on the Houston dataset. </a>


<p align="center">
<img src=/figure/DHSNet.emf width="80%">
</P>

<div align='center'>

**Figure 1. Framework of DHSnet.**

</div>
<br>



# 📝  Citation
If you find our paper helpful, please give a ⭐ and cite it as follows:
```
@ARTICLE{11265782,
  author={Liu, Rong and Liang, Junye and Yang, Jiaqi and Hu, Meiqi and He, Jiang and Zhu, Peng and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DHSNet: Dual Classification Head Self-Training Network for Cross-Scene Hyperspectral Image Classification}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Feature extraction;Adaptation models;Training;Kernel;Electronic mail;Measurement;Hyperspectral imaging;Land surface;Biological system modeling;Vectors;Central attention;cross-scene classification;domain adaptation (DA);hyperspectral image (HSI);self-training},
  doi={10.1109/TGRS.2025.3636101}}
  ```

# 📖 Relevant Projects

[1] <strong>Spectral Structure-Aware Initialization and Probability-Consistent Self-Training for Cross-Scene Hyperspectral Image Classification, GRSL, 2025</strong> | [Paper](https://ieeexplore.ieee.org/document/11020658)
<br><em>&ensp; &ensp; &ensp; Junye Liang, Jiaqi Yang, Rong Liu, Quanwei Liu, Peng Zhu</em>

[2] <strong>Hyper-LKCNet: Exploring the Utilization of Large Kernel Convolution for Hyperspectral Image Classification, JSTARS, 2025</strong> | [Paper](https://ieeexplore.ieee.org/abstract/document/11007459) | [Code](https://github.com/liurongwhm/Hyper-LKNet)
<br><em>&ensp; &ensp; &ensp; Rong Liu, Zhilin Li, Jiaqi Yang , Jian Sun, and Quanwei Liu</em>

# 🔩 Requirements
CUDA Version: 12.2
Python: 3.9
torch: 2.1.0

# 📚 Dataset
The dataset directory should look like this:
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
│   ├── paviaC.mat
│   └── paviaC_7gt.mat
└──  HyRANK
    ├── Dioni.mat
    └── Dioni_gt_out68.mat
    ├── Loukia.mat
    └── Loukia_gt_out68.mat

# 🔨Usage
1. Download and prepare your hyperspectral dataset. You can download Houston dataset in *./dataset/Houston*. 
2. Please change the source_name and target_name in train.py.
3. Run python train.py.
4. Default results directory is: *.*/*results*. You can check your classification maps here.


# 🎺 Statement
For any other questions please contact Junye Liang at [sysu.edu.cn](liangjy225@mail2.sysu.edu.cn).



