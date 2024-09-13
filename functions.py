import pandas as pd
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def read_dataset():
    vol = pd.read_csv('datasets/volume.csv', index_col=0, header=0)
    inf = pd.read_csv('datasets/information.csv', index_col=None, header=0)
    prc = pd.read_csv('datasets/price.csv', index_col=0, header=0)
    adj = pd.read_csv('datasets/adj.csv', index_col=0, header=0)  # check
    dis = pd.read_csv('datasets/new_data/A_function.csv', index_col=0, header=0)
    time = pd.read_csv('datasets/time.csv', index_col=None, header=0)

    occ = pd.read_csv('datasets/occupancy.csv', index_col=0, header=0)
    day = pd.read_csv('datasets/new_data/day_of_week.csv', index_col=0, header=0)
    fri = pd.read_csv('datasets/new_data/Fri_Sat.csv', index_col=0, header=0)
    hor = pd.read_csv('datasets/new_data/hour.csv', index_col=0, header=0)
    wea = pd.read_csv('datasets/new_data/wea.csv', index_col=0, header=0)
    fuc = pd.read_csv('datasets/new_data/A_function.csv', index_col=0, header=0)

    col = vol.columns
    vol = np.array(vol, dtype=float)
    cap = np.array(inf['count'], dtype=float).reshape(1, -1)  # parking_capability
    occ = np.array(occ, dtype=float) / cap
    prc = np.array(prc, dtype=float)
    adj = np.array(adj, dtype=float)
    dis = np.array(dis, dtype=float)

    day = np.array(day, dtype=float)
    fri = np.array(fri, dtype=float)
    hor = np.array(hor, dtype=float)
    wea = np.array(wea, dtype=float)
    fuc = np.array(fuc, dtype=float)
    time = pd.to_datetime(time, dayfirst=True)

    return vol, prc, adj, col, dis, cap, time, inf, occ, day, fri, hor, wea, fuc

# ---------data transform-----------
def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])
    return np.array(x), np.array(y)


def get_a_delta(adj):  # D^-1/2 * A * D^-1/2
    # adj.shape = np.size(node, node)
    deg = np.sum(adj, axis=0)
    deg = np.diag(deg)
    deg_delta = np.linalg.inv(np.sqrt(deg))
    a_delta = np.matmul(np.matmul(deg_delta, adj), deg_delta)
    return a_delta


def division(data, train_rate, valid_rate, test_rate):
    data_length = len(data)
    train_division_index = int(data_length * train_rate)
    valid_division_index = int(data_length * (train_rate + valid_rate))
    test_division_index = int(data_length * (1 - test_rate))
    train_data = data[:train_division_index, :]
    valid_data = data[train_division_index:valid_division_index, :]
    test_data = data[test_division_index:, :]
    return train_data, valid_data, test_data


def set_seed(seed, flag):
    if flag == True:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def metrics(test_pre, test_real):
    eps = 0.01
    MAPE_test_real = test_real
    MAPE_test_pre = test_pre
    MAPE_test_real[np.where(MAPE_test_real == 0)] = MAPE_test_real[np.where(MAPE_test_real == 0)] + eps
    MAPE_test_pre[np.where(MAPE_test_real == 0)] = MAPE_test_pre[np.where(MAPE_test_real == 0)] + eps
    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_real, test_pre)
    RAE = np.sum(abs(test_pre - test_real)) / np.sum(abs(np.mean(test_real) - test_real))
    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print(('RAE:{}'.format(RAE)))
    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2]
    return output_list


class CreateDataset(Dataset):
    def __init__(self, vol, prc, occ, day, fri, hor, wea, lb, pt, device, adj):  # adj
        vol, label = create_rnn_data(vol, lb, pt)
        prc, _ = create_rnn_data(prc, lb, pt)
        occ, _ = create_rnn_data(occ, lb, pt)
        day, _ = create_rnn_data(day, lb, pt)
        fri, _ = create_rnn_data(fri, lb, pt)
        hor, _ = create_rnn_data(hor, lb, pt)
        wea, _ = create_rnn_data(wea, lb, pt)

        self.vol = torch.Tensor(vol)
        self.prc = torch.Tensor(prc)
        self.label = torch.Tensor(label)

        self.occ = torch.Tensor(occ)
        self.day = torch.Tensor(day)
        self.fri = torch.Tensor(fri)
        self.hor = torch.Tensor(hor)
        self.wea = torch.Tensor(wea)

        self.device = device

    def __len__(self):
        return len(self.vol)

    def __getitem__(self, idx):  # vol: batch, seq, node
        output_vol = torch.transpose(self.vol[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_occ = torch.transpose(self.occ[idx, :, :], 0, 1).to(self.device)
        output_day = torch.transpose(self.day[idx, :, :], 0, 1).to(self.device)
        output_fri = torch.transpose(self.fri[idx, :, :], 0, 1).to(self.device)
        output_hor = torch.transpose(self.hor[idx, :, :], 0, 1).to(self.device)
        output_wea = torch.transpose(self.wea[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        return output_vol, output_prc, output_occ, output_day, output_fri, output_hor,  output_wea,  output_label

class CreateFastDataset(Dataset):
    def __init__(self, vol, prc, occ, day, fri, hor, wea, lb, pt, law, device, adj, num_layers=2, prob=0.6):  # adj
        vol, label = create_rnn_data(vol, lb, pt)
        prc, _ = create_rnn_data(prc, lb, pt)


        self.vol = torch.Tensor(vol)
        self.prc = torch.Tensor(prc)
        self.label = torch.Tensor(label)
        self.device = device
        self.adj = adj
        self.eye = torch.eye(adj.shape[0])
        self.deg = torch.sum(adj, dim=0)
        self.num_layers = num_layers
        self.law = -law

        # price
        chg = torch.randn(size=[self.vol.shape[2]]) / 2
        chg[torch.where(chg < prob)] = 0
        self.prc_chg = chg  # [node, ]

        # label
        chg = torch.unsqueeze(chg, dim=1)  # [node, 1]
        deg = torch.unsqueeze(self.deg, dim=1)  # [node, 1]
        label_chg = [-chg]
        hop_chg = chg
        for n in range(self.num_layers):  # graph propagation
            hop_chg = torch.matmul(self.adj - self.eye, hop_chg) * (1 / deg)
            label_chg.append(hop_chg)
        label_chg = torch.stack(label_chg, dim=1)  # [node, num_layers]
        label_chg = torch.sum(label_chg, dim=1)  # [node, ]
        self.label_chg = torch.squeeze(label_chg, dim=1)

    def __len__(self):
        return len(self.vol)

    def __getitem__(self, idx):  # vol: batch, seq, node
        # Pseudo Sampling
        prc_ch = torch.Tensor(self.prc[idx, :, :] * (1 + self.prc_chg))  # [node, seq]
        label_ch = torch.tan(torch.Tensor(self.label[idx, :] * (1 + self.label_chg / self.law)))  # [node, ]

        # to device
        output_vol = torch.transpose(self.vol[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        output_prc_ch = torch.transpose(prc_ch, 0, 1).to(self.device)
        output_label_ch = label_ch.to(self.device)
        return output_vol, output_prc, output_label, output_prc_ch, output_label_ch


class PseudoDataset(Dataset):
    def __init__(self, vol, prc, lb, pt, device, adj, law, num_layers=2, prop=0.4):  # adj
        vol, label = create_rnn_data(vol, lb, pt)
        prc, _ = create_rnn_data(prc, lb, pt)
        self.vol = torch.Tensor(vol)
        self.prc = torch.Tensor(prc)
        self.label = torch.Tensor(label)
        self.device = device
        self.adj = adj
        self.eye = torch.eye(adj.shape[0])
        self.deg = torch.sum(adj, dim=0)
        self.num_layers = num_layers
        self.prop = prop  # Proportion of nodes with price changes
        self.law = -law

        # price changes
        node_score = torch.rand(size=[self.vol.shape[2]])
        shred = torch.quantile(node_score, self.prop)
        prc_chg = torch.randn_like(node_score) / 2  # Percentage change in price
        prc_chg[torch.where(node_score > self.prop)] = 0
        self.prc_chg = prc_chg

        # label changes
        label_chg = self.law * prc_chg  # Percentage change in volumn
        label_chg = torch.unsqueeze(label_chg, dim=1)  # [node, 1]
        hop_chg = -label_chg
        label_chg = [label_chg]
        deg = torch.unsqueeze(self.deg, dim=1)  # [node, 1]
        for n in range(self.num_layers):  # graph propagation
            hop_chg = torch.matmul(self.adj - self.eye, hop_chg) * (1 / deg)
            label_chg.append(hop_chg)
        label_chg = torch.stack(label_chg, dim=1)  # [node, num_layers]
        label_chg = torch.sum(label_chg, dim=1)  # [node, ]
        self.label_chg = torch.squeeze(label_chg, dim=1)

    def __len__(self):
        return len(self.vol)

    def __getitem__(self, idx):  # vol: batch, seq, node
        # sampling
        pseudo_prc = torch.Tensor(self.prc[idx, :, :] * (1 + self.prc_chg))  # [node, seq]
        pseudo_label = torch.tan(torch.Tensor(self.label[idx, :] * (1 + self.label_chg)))  # [node, ]

        # to device
        output_vol = torch.transpose(self.vol[idx, :, :], 0, 1).to(self.device)
        output_prc = torch.transpose(self.prc[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        output_pseudo_prc = torch.transpose(pseudo_prc, 0, 1).to(self.device)
        output_pseudo_label = pseudo_label.to(self.device)

        return output_vol, output_prc, output_label, output_pseudo_prc, output_pseudo_label


def meta_division(data, support_rate, query_rate):
    data_length = len(data)
    support_division_index = int(data_length * support_rate)
    supprot_set = data[:support_division_index, :]
    query_set = data[support_division_index:, :]
    return supprot_set, query_set


def zero_init_global_gradient(model):
    grads = dict()
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        grads[name] = 0
    return grads


def data_mix(ori_data, pse_data, mix_ratio):
    shred = int(ori_data.shape[0] * mix_ratio)
    mix_data = ori_data
    mix_data[shred:] = pse_data[shred:]  # mix on the 1st dimension: batch
    return mix_data
