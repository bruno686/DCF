# Double Correction Framework for Denoising Recommendation

The code and datasets of our paper "Double Correction Framework for Denoising Recommendation"

# Overview

We propose a Double Correction Framework for Denoising Recommendation (DCF), which contains two correction components from views of more precise sample dropping and avoiding more sparse data. In the sample dropping correction component, we use the loss value of the samples over time to determine whether it is noise or not, increasing dropping stability. Instead of averaging directly, we use the damping function to reduce the bias effect of outliers. Furthermore, due to the higher variance exhibited by hard samples, we derive a lower bound for the loss through concentration inequality to identify and reuse hard samples. In progressive label correction, we iteratively re-label highly deterministic noisy samples and retrain them to further improve performance. Finally, extensive experimental results on three datasets and four backbones demonstrate the effectiveness and generalization of our proposed framework.

# Requirements

The model is implemented using PyTorch. The versions of packages used are shown below.

- numpy==1.18.0
- scikit-learn==0.22.1
- torch==1.6.1

# Data Preparation

The three data source we use comes from [Adressa](https://github.com/WenjieWWJ/DenoisingRec) , [Yelp](https://github.com/WenjieWWJ/DenoisingRec) and [MovieLens](https://github.com/wangyu-ustc/DeCA).

# Special thanks 
Very thanks to Dr.Wenjie Wang with his code [DenoisingRec](https://github.com/WenjieWWJ/DenoisingRec).

# Quick run

```js
python main.py
```
