import argparse
import json
import os
from copy import deepcopy
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.nn import DataParallel
from loader import MoleculeDataset
from model import GNN
from tqdm import tqdm

from contrastive import SupConLoss, compute_similarity, KContrastiveLoss, contrastive_loss
from splitters import scaffold_split, random_split


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='bbbp',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--filename', type=str, default='',
                        help='output filename')
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--aug1', type=str, default='none',
                        help='augmentation1')
    parser.add_argument('--aug2', type=str, default='reinforcement',
                        help='augmentation2')
    parser.add_argument('--max_step', type=int, default=10,
                        help='the max step of generating molecule')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs when performing k independent runs')
    parser.add_argument('--split', type=str, default='balanced_scaffold',
                        help="random or scaffold or balanced_scaffold")
    parser.add_argument('--loss', type=str, default='sup',
                        help="supervised contrastive learning or contrastive learning")
    args = parser.parse_args()

    torch.manual_seed(args.runseed)  # Set seeds for specific gpus so that the results are definitive and easy to reproduce
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)  # Set seeds for all gpus

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'mutag':
        num_tasks = 1
    elif args.dataset == 'lipo':
        num_tasks = 1
    elif args.dataset == 'qm7':
        num_tasks = 1
    elif args.dataset == 'qm8':
        num_tasks = 12
    elif args.dataset == 'qm9':
        num_tasks = 3
    else:
        raise ValueError("Invalid dataset name.")

    for n in range(args.num_runs):  # experiment with different seeds
        load_dir = 'model/' + args.dataset + '/pretrain/'
        save_dir = 'model/' + args.dataset + '/pretrain/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pretrained_model_str = 'supervised_' + args.dataset + '_seed_' + str(args.runseed) + '_' + str(args.seed)
        output_model_str = args.dataset + '_aug1_' + args.aug1 + '_aug2_' + args.aug2 + '_loss_' + args.loss + '_seed_' + str(args.runseed) + '_' + str(args.seed + n)
        pretrained_modelfile = load_dir + pretrained_model_str + '.pth'
        modelfile = save_dir + output_model_str + ".pth"
        txtfile = save_dir + output_model_str + ".txt"

        # set up model
        if os.path.exists(pretrained_modelfile):
            encoder = GNN(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                                  graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
            # if torch.cuda.device_count() > 1:
            #     encoder = DataParallel(encoder)
            encoder.to(device)
            wb = torch.load(pretrained_modelfile)
            encoder.load_state_dict(wb)
            print('load pretrained model successfully!')
        else:
            encoder = GNN(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio,
                          graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, pretrain=True)
            # if torch.cuda.device_count() > 1:
            #     encoder = DataParallel(encoder)
            encoder.to(device)

        dataset = MoleculeDataset("data/" + args.dataset, device=device, dataset_name=args.dataset)
        dataset.graph_pred = deepcopy(encoder)
        encoder.pretrain = True
        if args.split == "scaffold":
            smiles_list = dataset.smiles()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, frac_train=0.8,
                                                                        frac_valid=0.1, frac_test=0.1,
                                                                        seed=args.seed + n)
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, frac_train=0.8, frac_valid=0.1,
                                                                      frac_test=0.1, seed=args.seed + n)
            print("random")
        elif args.split == "balanced_scaffold":
            smiles_list = dataset.smiles()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, balanced=True,
                                                                        frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                                                                        seed=args.seed + n)
            print("balanced scaffold")
        else:
            raise ValueError("Invalid split option.")

        # set up optimizer
        model_param_group = []
        # model_param_group.append({"params": encoder.gnn.parameters(), "lr": args.lr})
        # model_param_group.append({"params": encoder.proj_head.parameters(), "lr": args.lr})
        optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.decay)

        # 记录对比学习损失
        with open(txtfile, "w") as myfile:
            myfile.write('epoch: train_loss\n')

        loss_curve = []  # record the curve of contrastive loss
        for epoch in tqdm(range(1, args.epochs + 1)):
            print("====epoch " + str(epoch))
            dataset = train_dataset.shuffle()
            dataset1 = deepcopy(dataset)
            dataset1.aug = args.aug1
            dataset.aug = args.aug2
            loader1 = DataLoader(dataset1, args.batch_size, shuffle=True)
            loader = DataLoader(dataset, args.batch_size, shuffle=True)

            total_loss = 0
            encoder.train()
            for data1, data in tqdm(zip(loader1, loader)):
                optimizer.zero_grad()
                # encoder.predict_f = False

                if args.aug2 == 'reinforcement':
                    temp = data[0]
                    data = temp
                data1.to(device)
                data.to(device)

                out1 = encoder(data1)  # 原样本
                out = encoder(data)  # 困难正样本
                # cl = ContrastiveLoss()
                cl = SupConLoss(device=device, temperature=0.1)
                # cl = ContrastiveLoss(device)
                feature = torch.stack([out1, out], dim=1)
                labels = data1.y

                cl.to(device)
                if args.loss == 'sup':
                    mask = compute_similarity(labels)  # 计算相似度
                    loss = cl(feature, mask=mask)
                else:
                    # loss = cl(out1, out)
                    # loss = contrastive_loss(out1, out)
                    loss = cl(feature)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            encoder.save_pretrained(modelfile)
            train_loss = total_loss
            print("train loss: %f" % (train_loss))
            loss_curve.append(train_loss)
            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ': ' + str(train_loss) + "\n")

        if not output_model_str == "":  # save the pre-trained model after contrastive learning
            x = range(1, epoch + 1)
            y = loss_curve
            plt.plot(x, y)
            plt.xlabel('epoch')
            plt.ylabel('pretrain loss')
            plt.savefig(save_dir + output_model_str + '_pretrain_loss.jpg')


if __name__ == "__main__":
    main()
