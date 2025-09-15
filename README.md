<div align="center">

# SWiFT: Soft-Mask Weight Fine-tuning for Bias Mitigation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2508.18826-B31B1B.svg)](https://doi.org/10.59275/j.melba.2025-de23)
[![Conference](http://img.shields.io/badge/MELBA-2025-4b44ce.svg)](https://www.melba-journal.org/issues/faimi25.html)

</div>

## Description

Recent studies have shown that Machine Learning (ML) models can exhibit bias in real-world scenarios, posing significant challenges in ethically sensitive domains such as healthcare. Such bias can negatively affect model fairness, model generalization abilities and further risks amplifying social discrimination. There is a need to remove biases from trained models. Existing debiasing approaches often necessitate access to original training data and need extensive model retraining; they also typically exhibit trade-offs between model fairness and discriminative performance. To address these challenges, we propose Soft-Mask Weight Fine-Tuning (SWiFT), a debiasing framework that efficiently improves fairness while preserving discriminative performance with much less debiasing costs. Notably, SWiFT requires only a small external dataset and only a few epochs of model fine-tuning. The idea behind SWiFT is to first find the relative, and yet distinct, contributions of model parameters to both bias and predictive performance. Then, a two-step fine-tuning process
updates each parameter with different gradient flows defined by its contribution. Extensive experiments with three bias sensitive attributes (gender, skin tone, and age) across four dermatological and two chest X-ray datasets demonstrate that SWiFT can consistently reduce model bias while achieving competitive or even superior diagnostic accuracy under common fairness and accuracy metrics, compared to the state-of-the-art. Specifically, we demonstrate improved model generalization ability as evidenced by superior performance on several out-of-distribution (OOD) datasets.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/vios-s/SWiFT.git
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/vios-s/SWiFT.git
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```
## Dataset

The datasets used in this project are available from the following sources (registration may be required):

| Dataset                          | Access                                                                 |
|----------------------------------|------------------------------------------------------------------------|
| ISIC 2020                        | [Link](https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256)           |
| ISIC 2019/2018/2017              | [Link](https://www.kaggle.com/cdeotte/jpeg-isic2019-256x256)           |
| Interactive Atlas of Dermoscopy  | [Link](https://derm.cs.sfu.ca/Welcome.html)                            |
| PAD                              | [Link](https://data.mendeley.com/datasets/zr7vgbcyr2/1)                 |
| Fitzpatrick17k                   | [Link](https://github.com/mattgroh/fitzpatrick17k)                     |
| MIMIC-CXR                        | [Link](https://physionet.org/content/mimic-cxr/2.1.0/)                 |
| CheXpert                         | [Link](https://stanfordmlgroup.github.io/competitions/chexpert/)       |
| NIH ChestX-ray14                 | [Link](https://www.kaggle.com/datasets/nih-chest-xrays/data)           |


## How to run

# Run the debiasing algorithm
```bash
python algorithm/TwoStageFinetune.py
```
You can overide any parameter in argument.py or from command line like this 

```bash
python algorithm/TwoStageFinetune.py --arch resnet50 --task 'xray' --attr 'age_attribute' --lr-base 0.000001 --lr-forget 0.000001 --beta 0.01 --model-dir './logs/model/resnet50_mimic_val0_gender.ckpt' --csv-dir './data/chestXray/csv/mimic_val_gender_0.csv' --batch-size 128 --num-attr 'binary'
```

# Train the baseline model
Pre-train the baseline model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```







