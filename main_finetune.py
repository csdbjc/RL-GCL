import argparse
import logging
import math
import os

from torch import optim
from torch import nn
from torch_geometric.nn import DataParallel

from random import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, average_precision_score, f1_score, \
    recall_score, precision_score
from tqdm import tqdm

from cite.scheduler import build_lr_scheduler
from splitters import scaffold_split, random_split

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader, DataListLoader
from model import GNN


def train_cls(model, device, loader, optimizer, scheduler, criterion):  # train model for classification tasks
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        optimizer.zero_grad()
        # y = [data.y for data in batch]
        # y = torch.cat(y).to(device)
        y = batch.y
        is_valid = (y == y).float()
        # loss = criterion(pred.to(torch.float64)[is_valid], batch.y.to(torch.float64)[is_valid])  # calculate loss
        loss = criterion(pred.to(torch.float64) * is_valid, torch.nan_to_num(y, nan=0.0) * is_valid)
        loss = loss.sum() / is_valid.sum()

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss


def train_reg(model, device, loader, optimizer, scheduler, criterion):  # train model for regression tasks
    model.train()
    total_loss = 0
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        optimizer.zero_grad()
        # y = [data.y for data in batch]
        # y = torch.cat(y).to(device)
        y = batch.y
        loss = criterion(pred.to(torch.float64), y)  # MSE loss
        loss = loss / y.shape[1]
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        y_true.append(y)
        y_scores.append(pred)
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()

    return total_loss

def evaluate_cls(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    y_pred = np.where(y_score >= 0.5, 1, 0)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return auc, aupr, f1, recall, precision

def evaluate_reg(y_true, y_score):
    rmse = math.sqrt(mean_squared_error(y_true, y_score))
    # rmse = mean_squared_error(y_true, y_score)
    mae = mean_absolute_error(y_true, y_score)
    return rmse, mae


def eval_cls(model, device, loader):  # evaluate model for classification tasks
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        # y = [data.y for data in batch]
        # y = torch.cat(y).to(device)
        y = batch.y
        y_true.append(y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    metric_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            islabeled = y_true[:, i] == y_true[:, i]
            is_valid = np.argwhere(islabeled == True)
            metric_list.append(evaluate_cls(y_true[is_valid, i], y_scores[is_valid, i]))
        else:
            metric_list.append((float('nan'),) * 5)
    if len(metric_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(metric_list)) / y_true.shape[1]))
    return np.nanmean(metric_list, axis=0)  # y_true.shape[1]


def eval_reg(model, device, loader):  # evaluate model for regression tasks
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)
        # y = [data.y for data in batch]
        # y = torch.cat(y).to(device)
        y = batch.y
        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    metric_list = []
    for i in range(y_true.shape[1]):
        metric_list.append(evaluate_reg(y_true[:, i], y_scores[:, i]))
    return np.mean(metric_list, axis=0)


def main():
    # model parameter
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

    # data parameter
    parser.add_argument('--dataset', type=str, default='bbbp',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--filename', type=str, default='',
                        help='output filename')
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default='balanced_scaffold',
                        help="random or scaffold or balanced_scaffold")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='none',
                        help='augmentation1')
    parser.add_argument('--aug2', type=str, default='reinforcement',
                        help='augmentation2')
    parser.add_argument('--loss', type=str, default='sup',
                        help="supervised contrastive learning or contrastive learning")
    parser.add_argument('--dataset_load', type=str, default='',
                        help='load pretrain model from which dataset.')
    parser.add_argument('--protocol', type=str, default='linear',
                        help='downstream protocol, linear, nonlinear')
    parser.add_argument('--semi_ratio', type=float, default=1.0,
                        help='proportion of labels in semi-supervised settings')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs when performing k independent runs')

    # other
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    args = parser.parse_args()

    # print parameter
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(message)

    torch.manual_seed(args.runseed)  # model seed
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")  # device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
        task_type = 'cls'
    else:
        task_type = 'reg'

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

    # set up dataset
    dataset = MoleculeDataset("data/" + args.dataset, device=device, dataset_name=args.dataset)
    print(dataset.data)

    best = []
    for n in range(args.num_runs):  # experiment with different seeds
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

        # semi-supervised settings
        if args.semi_ratio != 1.0:
            n_total, n_sample = len(train_dataset), int(len(train_dataset) * args.semi_ratio)
            print('sample {:.2f} = {:d} labels for semi-supervised training!'.format(args.semi_ratio, n_sample))
            all_idx = list(range(n_total))
            random.seed(args.seed + n)
            idx_semi = random.sample(all_idx, n_sample)
            train_dataset = train_dataset[torch.tensor(idx_semi)]  #int(len(train_dataset)*args.semi_ratio)
            print('new train dataset size:', len(train_dataset))
        else:
            print('finetune using all data!')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if not args.dataset_load == "":
            load_dir = 'model/' + args.dataset_load + '/pretrain/'
        else:
            load_dir = 'model/' + args.dataset + '/pretrain/'
        save_dir = 'model/' + args.dataset + '/finetune/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(load_dir):
            os.makedirs(load_dir)

        if not args.dataset_load == "":
            input_model_str = (args.dataset_load + '_aug1_' + args.aug1 + '_aug2_' + args.aug2 + '_loss_' + args.loss +
                               '_seed_' + str(args.runseed) + '_' + str(args.seed + n))
            output_model_str = args.dataset + '_semi_' + str(
                args.semi_ratio) + '_protocol_' + args.protocol + '_seed_' + str(args.runseed) + '_' + str(
                args.seed + n)
        else:
            input_model_str = ""  # no pretrain
            output_model_str = 'supervised_' + args.dataset + '_seed_' + str(args.runseed) + '_' + str(
                args.seed + n)

        txtfile = save_dir + output_model_str + ".txt"
        modelfile = load_dir + output_model_str + ".pth"  # 当为有监督时，存储图编码器

        # set up model
        model = GNN(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                    graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        # model = DataParallel(model)
        model.predict_f = True

        # model = build_model(dataset.f_atoms.shape[1], dataset.f_bonds.shape[1], args.emb_dim, num_tasks, device)
        # initialize_weights(model)
        if not input_model_str == "":
            wb = torch.load(load_dir + input_model_str + '.pth')
            model.load_state_dict(wb)
            print('successfully load pretrained model!')
        else:
            print('No pretrain! Supervised learning!')
        model.to(device)

        # set up optimizer
        # if linear protocol, fix GNN layers
        model_param_group = []
        # if args.graph_pooling == "attention":
        #     model_param_group.append({"params": model.module.pool.parameters(), "lr": args.lr})
        if args.protocol == 'nonlinear':
            model_param_group.append({"params": model.parameters(), "lr": args.lr})
            print("finetune protocol, train all the layers!")
        elif args.protocol == 'linear':
            model_param_group.append({"params": model.pred.parameters(), "lr": args.lr})
            print("linear protocol, only train the top layer!")
            for name, param in model.named_parameters():  # 冻结参数
                if not 'pred' in name:
                    param.requires_grad = False
        else:
            raise ValueError("Invalid protocol!")
        optimizer = optim.Adam(model_param_group)
        scheduler = build_lr_scheduler(optimizer, args, len(train_dataset))

        # all task info summary
        print('=========task summary=========')
        print('Dataset: ', args.dataset)
        if args.semi_ratio == 1.0:
            print('full-supervised {:.2f}'.format(args.semi_ratio))
        else:
            print('semi-supervised {:.2f}'.format(args.semi_ratio))
        if input_model_str == '':
            print('scratch or finetune: scratch')
            print('loaded model from: - ')
        else:
            print('scratch or finetune: finetune')
            print('loaded model from: ', args.dataset_load)
        print('Protocol: ', args.protocol)
        print('task type:', task_type)
        print('=========task summary=========')
        # training based on task type
        if task_type == 'cls':
            with open(txtfile, "w") as myfile:
                myfile.write("Epoch,Train_AUC,Train_AUPR,Train_F1,Train_Recall,Train_Precision,Valid_AUC,Valid_AUPR,Valid_F1,Valid_Recall,Valid_Precision,Test_AUC,Test_AUPR,Test_F1,Test_Recall,Test_Precision\n")
            best_valid_metric = [None] * 5
            best_test_metric = [None] * 5
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            for epoch in range(1, args.epochs + 1):
                print("====epoch " + str(epoch))
                loss = train_cls(model, device, train_loader, optimizer, scheduler, criterion)
                print('loss:', loss)

                print("====Evaluation")
                train_metric = eval_cls(model, device, train_loader)
                val_metric = eval_cls(model, device, val_loader)
                test_metric = eval_cls(model, device, test_loader)

                with open(txtfile, "a") as myfile:
                    myfile.write(
                        f"{epoch},{train_metric[0]:.4f},{train_metric[1]:.4f},{train_metric[2]:.4f},{train_metric[3]:.4f},{train_metric[4]:.4f},")
                    myfile.write(
                        f"{val_metric[0]:.4f},{val_metric[1]:.4f},{val_metric[2]:.4f},{val_metric[3]:.4f},{val_metric[4]:.4f},")
                    myfile.write(
                        f"{test_metric[0]:.4f},{test_metric[1]:.4f},{test_metric[2]:.4f},{test_metric[3]:.4f},{test_metric[4]:.4f}\n")

                # 根据验证集的指标选择最优的测试集指标
                for i in range(5):
                    if best_valid_metric[i] is None or val_metric[i] > best_valid_metric[i]:
                        best_valid_metric[i] = val_metric[i]
                        best_test_metric[i] = test_metric[i]

                print("train auc: %.3f val auc: %.3f test auc: %.3f" % (train_metric[0], val_metric[0], test_metric[0]))
                print("train aupr: %.3f val aupr: %.3f test aupr: %.3f" % (train_metric[1], val_metric[1], test_metric[1]))
                print("train f1: %.3f val f1: %.3f test f1: %.3f" % (train_metric[2], val_metric[2], test_metric[2]))
                print("train recall: %.3f val recall: %.3f test recall: %.3f" % (train_metric[3], val_metric[3], test_metric[3]))
                print("train precision: %.3f val precision: %.3f test precision: %.3f" % (train_metric[4], val_metric[4], test_metric[4]))

            print('seed {:d} => best val auc {:.4f} and best test auc {:.4f}'.format(args.seed + n, best_valid_metric[0],
                                                                                     best_test_metric[0]))
            best.append(best_test_metric)
            model.save_pretrained(modelfile)
            metrics_names = ["AUC", "AUPR", "F1", "Recall", "Precision"]
            mean_metrics = np.mean(best, axis=0)  # mean
            std_metrics = np.std(best, axis=0)  # std
            with open(txtfile, "a") as myfile:
                for i, name in enumerate(metrics_names):
                    print(f"{name}: {mean_metrics[i]:.4f} ± {std_metrics[i]:.4f}")
                    myfile.write(
                        f"{name}: {mean_metrics[i]:.4f} ± {std_metrics[i]:.4f}")
        elif task_type == 'reg':
            with open(txtfile, "w") as myfile:
                myfile.write(
                    "Epoch,Train_RMSE,Train_MAE,Valid_RMSE,Valid_MAE,Test_RMSE,Test_MAE\n")
            best_valid_metric = [np.Inf] * 2
            best_test_metric = [np.Inf] * 2
            criterion = torch.nn.MSELoss()
            for epoch in range(1, args.epochs + 1):
                print("====epoch " + str(epoch))
                loss = train_reg(model, device, train_loader, optimizer, scheduler, criterion)
                print('loss:', loss)
                print("====Evaluation")

                train_metric = eval_reg(model, device, train_loader)
                val_metric = eval_reg(model, device, val_loader)
                test_metric = eval_reg(model, device, test_loader)

                with open(txtfile, "a") as myfile:
                    myfile.write(
                        f"{epoch},{train_metric[0]:.4f},{train_metric[1]:.4f},")
                    myfile.write(
                        f"{val_metric[0]:.4f},{val_metric[1]:.4f},")
                    myfile.write(
                        f"{test_metric[0]:.4f},{test_metric[1]:.4f}\n")

                print("train rmse: %.3f val rmse: %.3f test rmse: %.3f" % (train_metric[0], val_metric[0], test_metric[0]))
                print("train mae: %.3f val mae: %.3f test mae: %.3f" % (train_metric[1], val_metric[1], test_metric[1]))

                # Early stopping
                for i in range(2):
                    if val_metric[i] < best_valid_metric[i]:
                        best_valid_metric[i] = val_metric[i]
                        best_test_metric[i] = test_metric[i]

            print('seed {:d} => with best val rmse: {:.4f} and test rmse: {:.4f}'.format(args.seed + n, best_valid_metric[0],
                                                                                         best_test_metric[0]))
            best.append(best_test_metric)
            model.save_pretrained(modelfile)
            metrics_names = ["RMSE", "MAE"]
            mean_metrics = np.mean(best, axis=0)  # mean
            std_metrics = np.std(best, axis=0)  # std
            with open(txtfile, "a") as myfile:
                for i, name in enumerate(metrics_names):
                    print(f"{name}: {mean_metrics[i]:.4f} ± {std_metrics[i]:.4f}")
                    myfile.write(
                        f"{name}: {mean_metrics[i]:.4f} ± {std_metrics[i]:.4f}")


if __name__ == "__main__":
    main()
