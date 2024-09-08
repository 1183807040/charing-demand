import copy
import baselines
import torch
import numpy as np
import pandas as pd
import functions as fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
import learner

# system configuration
use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2024, flag=True)

# hyper params
model_name = 'HSSTM'  # TODO: NI
seq_l = 12
pre_l = 6
bs = 64
# p_epoch = 200
n_epoch = 2
# law_list = np.array([-1.48, -0.74])  # price elasticities of demand for EV charging. Recommend: up to 5 elements.
is_train = True
mode = 'completed'  # 'simplified' or 'completed'
# is_pre_train = False

# input data
vol, prc, adj, col, dis, cap, time, inf, occ, day, fri, hor, wea = fn.read_dataset()
adj_dense = torch.Tensor(adj)
adj_dense_cuda = adj_dense.to(device)
adj_sparse = adj_dense.to_sparse_coo().to(device)

# dataset division
train_volumn, valid_volumn, test_volumn = fn.division(vol, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_price, valid_price, test_price = fn.division(prc, train_rate=0.6, valid_rate=0.2, test_rate=0.2)


train_occ, valid_occ, test_occ = fn.division(occ, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_day, valid_day, test_day = fn.division(day, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_fri, valid_fri, test_fri = fn.division(fri, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_hor, valid_hor, test_hor = fn.division(hor, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_wea, valid_wea, test_wea = fn.division(wea, train_rate=0.6, valid_rate=0.2, test_rate=0.2)

# data
train_dataset = fn.CreateDataset(train_volumn, train_price, train_occ, train_day, train_fri, train_hor, train_wea, seq_l, pre_l, device, adj_dense)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
valid_dataset = fn.CreateDataset(valid_volumn, valid_price, valid_occ, valid_day, valid_fri, valid_hor, valid_wea, seq_l, pre_l, device, adj_dense)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_volumn), shuffle=False)  # TODO
test_dataset = fn.CreateDataset(test_volumn, test_price, test_occ, test_day, test_fri, test_hor, test_wea, seq_l, pre_l, device, adj_dense)
test_loader = DataLoader(test_dataset, batch_size=len(test_volumn), shuffle=False)

# training setting
# model = models.PAG(a_sparse=adj_sparse).to(device)  # init model
# model = FGN().to(device)
# model = baselines.VAR(247, seq_l,2)
# model = baselines.LSTM(seq_l, 2, node=247)
# model = baselines.LstmGcn(seq_l, 2, adj_dense_cuda)
# model = baselines.LstmGat(seq_l, 7, adj_dense_cuda, adj_sparse)
model = baselines.HSTGCN(seq_l, 2, adj_dense_cuda, adj_dense_cuda)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)
loss_function = torch.nn.MSELoss()
valid_loss = 100000

if is_train is True:
    model.train()
    # if is_pre_train is True:
    #     if mode == 'simplified':  # a simplified way of physics-informed meta-learning
    #         model = learner.fast_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l,
    #                                       pre_l, device, adj_dense)
    #
    #     elif mode == 'completed':  # the completed process
    #         model = learner.physics_informed_meta_learning(law_list, model, model_name, p_epoch, bs, train_occupancy,
    #                                                        train_price, seq_l, pre_l, device, adj_dense)
    #     else:
    #         print("Mode error, skip the pre-training process.")

    for epoch in tqdm(range(n_epoch), desc='Fine-tuning'):
        for j, data in enumerate(train_loader):
            '''
            volumn = (batch,  node, seq)  #reviserd
            price = (batch, node ,seq )
            label = (batch, node)
            '''
            model.train()
            volumn, price, occ, day, fri, hor, wea, label = data
            optimizer.zero_grad()
            predict = model(volumn, price) # , occ, day, fri, hor, wea
            loss = loss_function(predict, label)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        for j, data in enumerate(valid_loader):
            '''
            volumn = (batch, seq, node)
            price = (batch, seq, node)
            label = (batch, node)
            '''

            model.train()
            volumn, price, occ, day, fri, hor, wea, label = data
            predict = model(volumn, price) # , occ, day, fri, hor, wea
            loss = loss_function(predict, label)
            if loss.item() < valid_loss:
                valid_loss = loss.item()
                torch.save(model,
                           './checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + mode + '.pt')

model = torch.load('./checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + mode + '.pt')
# test
model.eval()
result_list = []
predict_list = np.zeros([1, adj_dense.shape[1]])
label_list = np.zeros([1, adj_dense.shape[1]])
for j, data in enumerate(test_loader):
    volumn, price, occ, day, fri, hor, wea, label = data  # volumn.shape = [batch, seq, node]
    print('volumn:', volumn.shape, 'price:', price.shape, 'label:', label.shape)
    with torch.no_grad():
        predict = model(volumn, price) # , occ, day, fri, hor, wea
        predict = predict.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        predict_list = np.concatenate((predict_list, predict), axis=0)
        label_list = np.concatenate((label_list, label), axis=0)

output_no_noise = fn.metrics(test_pre=predict_list[1:, :], test_real=label_list[1:, :])
result_list.append(output_no_noise)
result_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'], data=result_list)
result_df.to_csv('./results' + '/' + model_name + '_' + str(pre_l) + 'bs' + str(bs) + '.csv', encoding='gbk')
