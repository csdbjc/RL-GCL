# RL-GCL
This repository is the official implementation of `
RL-GCL`, 
which is model proposed in a paper: 
## Brief introduction
**a. Graph Augmentation with Reinforcement.** We want to obtain the 
corresponding hard molecular augmentation based on the original molecule 
by minimizing molecular similarity and label difference. Four different 
modification operations are performed iteratively, and the final molecule 
with the highest reward is selected.  
**b. Supervised Contrastive Learning.** We use the original and augmented molecular graphs to select 
positive sample pairs with consistent labels and negative sample pairs 
with inconsistent labels. The graph encoder is then trained to maximize 
the consistency of label-consistent graph views and mitigate the impact 
of false negative samples.  
**c. Fine-tuning.** A classifier is 
connected downstream and trained using two different fine-tuning 
strategies.
![](/picture/overview.png "")
## Overview
This project mainly contains the following parts.  

    ├── cite                            # cite code from other papers
    ├── data                            # store the molecular datasets for pre-training and fine-tuning
    │   ├── bace.csv                    # downstream dataset BACE
    │   ├── bbbp.csv                    # downstream dataset BBBP
    │   ├── clintox.csv                 # downstream dataset ClinTox
    │   ├── esol.csv                    # downstream dataset ESOL
    │   ├── freesolv.csv                # downstream dataset FreeSolv
    │   ├── hiv.csv                     # downstream dataset HIV
    │   ├── lipo.csv                    # downstream dataset Lipophilicity
    │   ├── muv.csv                     # downstream dataset MUV
    │   ├── qm7.csv                     # downstream dataset QM7
    │   ├── qm8.csv                     # downstream dataset QM8
    │   ├── qm9.csv                     # downstream dataset QM9
    │   ├── sider.csv                   # downstream dataset SIDER
    │   ├── tox21.csv                   # downstream dataset Tox21
    │   ├── toxcast.csv                 # downstream dataset ToxCast
    │   └── zinc15_250K.csv             # pre-train dataset ZINC250K
    ├── model                           # store the training log and checkpoints of the model
    ├── contrastive.py                  # supervised Comparative Learning Loss Function
    ├── deep_q_networks.py              # the structure of deep Q-networks
    ├── environment.py                  # reinforment learning environment
    ├── train.py                        # train code for fine-tuning
    ├── finetune.sh                     # conduct fine-tuning
    ├── loader.py                       # load data
    ├── main_cl.py                      # pre-train model through contrastive learning
    ├── main_finetune.py                # semi-supervised and linear fine-tuning
    ├── model.py                        # GNN model
    ├── mol_function.py                 # calculate molecular similarity
    └── splitters.py                    # split dataset
# Quick start
If you want to use our pre-trained model directly for 
molecular property prediction, please run the following 
command:

    >> bash finetune.sh
| Parameter   | Description              |
|-------------|--------------------------|
| dataset     | Pre-trained task dataset |
| dataset_load | Downstream task dataset  |
| protocol    | Fine-tuning protocols    |
# Step-by-step guidelines
### Genetare hard molecular graph augmentations
You can download the generated molecular graph augmentations 
[**here**](https://drive.google.com/file/d/10QImFrfVSzDrvrWyKALRwt0bsOis0ZCl/view?usp=drive_link) and place them in RL-GCL to pre-train a new model by 
contrastive learning, or you can skip this step and 
**Graph contrastive learning** to perform **Fine-tuning** step 
directly using our pre-trained model.
### Graph contrastive learning
We use the datasets in MoleculeNet for comparative learning of pre-trained models. The pre-trained models on the corresponding datasets are obtained by executing the following code.
    
    >> python main_cl.py --dataset bbbp --loss sup --seed 0
| Parameter | Description                    |
|-----------|--------------------------------|
| dataset   | Pre-trained task dataset       |
| loss      | Contrastive loss function      |
| seed      | Seed for splitting the dataset |
### Fine-tuning 
We use two different protocols for downstream fine-tuning.

If you want to use **linear fine-tuning protocol**.

    >> python main_finetune.py --dataset bbbp --dataset_load bbbp --protocol linear --seed 0
If you want to use **semi-supervised fine-tuning protocol**.

    >> python main_finetune.py --dataset bbbp --dataset_load bbbp --protocol nonlinear --seed 0
If you want to use **supervised learning**。

    >> python main_finetune.py --dataset bbbp --protocol nonlinear --seed 0

| Parameter | Description                    |
|----------|--------------------------------|
| dataset  | Downstream task dataset        |
| loss     | Pre-trained task dataset       |
| protocol | Fine-tuning protocols          |
| seed     | Seed for splitting the dataset |

