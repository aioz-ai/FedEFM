# FedEFM: Federated Endovascular Foundation Model with Unseen Data

> In endovascular surgery, the precise identification of catheters and guidewires in X-ray images is essential for reducing intervention risks. However, accurately segmenting catheter and guidewire structures is challenging due to the limited availability of labeled data. Foundation models offer a promising solution by enabling the collection of similar-domain data to train models whose weights can be fine-tuned for downstream tasks. Nonetheless, large-scale data collection for training is constrained by the necessity of maintaining patient privacy. This paper proposes a new method to train a foundation model in a decentralized federated learning setting for endovascular intervention. To ensure the feasibility of the training, we tackle the unseen data issue using differentiable Earth Mover's Distance within a knowledge distillation framework. Once trained, our foundation model's weights provide valuable initialization for downstream tasks, thereby enhancing task-specific performance. Intensive experiments show that our approach achieves new state-of-the-art results, contributing to advancements in endovascular intervention and robotic-assisted endovascular surgery, while addressing the critical issue of data sharing in the medical domain.


## Description

Implementation of our paper [FedEFM: Federated Endovascular Foundation Model with Unseen Data](). 

## Table of Contents
  * [Description](#description)
  * [Table of Contents](#table-of-contents)
  * [Federated Foundation Model training](#federated-foundation-model-training)
  * [Downstream Tasks](#downstream-tasks)
  * [Acknowleagments](#acknowleagments)


## Federated Foundation Model training

### Dependencies 

Please install all the dependence packages by the following command:
```
pip install -r requirements.txt
```

### Dataset 

We use the following dataset for training the foundation model: [VESSEL12](https://paperswithcode.com/dataset/vessel12), [DRIVE](https://drive.grand-challenge.org/), [SenNet](https://www.kaggle.com/competitions/blood-vessel-segmentation/data), [Medical Decathlon](http://medicaldecathlon.com/), and our private data obtained from hospitals and laboratories in the UK (UoL Endovascular Simulation and UoL Catheter & Guidewire). The dataset can be downloaded via the following [link](https://vision.aioz.io/f/1ade5bb38eb445b0bee0/?dl=1) 

After downloading, you will obtain the folder `med`, please put it in `data` folder. 
### Training 

Run the following command to train the foundation model: 

```
bash train_ffl.sh
```
You can modify the hyperparamaters in the `train_ffl.sh` for various setups: 

- `--network_name`: DFL network. Only Gaia network is supported currently. 
- `--architecture`: Network topology. Can be one of the following setup: `ring`, `matcha`.
- `--n_round`: Number of communication rounds need to be training. 
- `--bz_train`: Batch size for local training in each silo. 
- `--local_steps`: Number of iterations performed in local training before moving to the communication phase in a communication round.

After several (or all) communication rounds, the model will be saved in `logg` folder, including the model of each silo, the global model, and the global model with the best performance in test set.

## Downstream Tasks 

### Segmentation tasks (TransUNet) 

First, please navigate to the folder contains the implementation: 
```
cd Downstream/TransUNet
```
#### Dependencies 

Install all the packages by running this command: 

```
pip install -r requirements.txt
```

#### Dataset 

We use the private dataset obtained from hospitals and laboratories in UK to evaluate downstream segmentation task. Please download the dataset [here](https://vision.aioz.io/f/ec5e0da6f18d4e938b1d/). 

After downloading, you will get the folder `data` contains phantom, animal, and simulation dataset. 

#### Checkpoint

You can download the checkpoint of foundation model [here](https://vision.aioz.io/f/a1a996c76d8b4a8c9166/). After downloading, you will get the folder `checkpoints` that contains checkpoint for TransUNet ViT backbone.
#### Training 

Run the following command to train the model: 

```
python3 train.py --max_epoch 100 --batch_size 16 --base_lr 0.01 --img_size 256 --vit_name R50-ViT-B_16 --dataset_domain animal
```

The following command will run setup for Animal dataset. If you want to run the code on Phantom or Simulation dataset, just change the  `--dataset_domain` argument to `phantom` or `sim` respectively. 

Currenly, our pretrained foundation model only supports `R50-ViT-B-16` backbone of TransUNet. 


## Acknowleagments 

- For training foundation model, we follow [this implementation](https://github.com/omarfoq/communication-in-cross-silo-fl?tab=readme-ov-file) in decentralized federated learning setup. We thank the authors for sharing the code.
- For downstream tasks, we followed the [TransUNet implementation](https://github.com/Beckschen/TransUNet) in intergrating our foundation model to TransUNet. We thank the authors for sharing the code.
