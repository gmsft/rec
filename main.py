import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from args import get_args
from models.AutoRec import AutoRec
from models.MF import MF
from utils.data_process import load_and_split, Dataset_ml, get_rating_matrix, Dataset_mat


def train_AutoRec(epoch,lr,weight_decay, latent_dim, num_users, device, dropout):
    model_autoRec = AutoRec(num_users, latent_dim, dropout).to(device)
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model_autoRec.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epoch):
        loss_train = []
        loss_test = []
        for e in mat_dl:
            optim.zero_grad()
            e = e.to(dtype=torch.float, device=device)
            out = model_autoRec(e)
            loss_ = loss(e, out)
            loss_.backward()
            optim.step()
            loss_train.append(loss_.item())
        with torch.no_grad():
            for tst in test_mat_dl:
                tst = tst.to(dtype=torch.float, device=device)
                # print(f'tst={tst[0]}')
                test_out = model_autoRec(tst)
                # print(f'test_out={test_out}')
                test_loss = loss(test_out, tst)
                loss_test.append(test_loss.item())
        print(f'epoch {ep}: loss = {np.mean(loss_train)}, test_loss={np.mean(loss_test)}')


def train_MF(epoch, lr, weight_decay, num_users, num_items, latent_dim, device):
    model = MF(num_users, num_items, latent_dim).to(device)
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for e in range(epoch):
        for u, i, r, t in train_dl:
            u, i = u.to(device), i.to(device)
            r = r.to(device, dtype=torch.float)
            out = model(u, i)
            loss_ = loss(out, r)
            optim.zero_grad()
            loss_.backward()
            optim.step()
        with torch.no_grad():
            test_out = model(torch.tensor(test_data['user'].tolist(), device=device),
                             torch.tensor(test_data['item'].tolist(), device=device))
            test_loss = loss(test_out, torch.tensor(test_data['rating'].tolist(), device=device, dtype=torch.float))
        print(f'epoch {e}: loss = {loss_}, test_loss={test_loss}')


# init parameters
args = get_args()
csv_path = args.csv_path
device = args.device
epoch = args.epoch
latent_dim = args.latent_dim
weight_decay = args.weight_decay
batch_size=args.batch_size
dropout = args.dropout
lr = args.lr
# load data
train_data, test_data, num_users, num_items = load_and_split(csv_path, implicit=False)
train_ds = Dataset_ml(train_data)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
r_mat = get_rating_matrix(train_data, num_users, num_items, implicit=False)
test_mat = get_rating_matrix(test_data, num_users, num_items, implicit=False)

mat_ds = Dataset_mat(r_mat)
mat_dl = DataLoader(mat_ds, batch_size=batch_size)
test_mat_ds = Dataset_mat(test_mat)
test_mat_dl = DataLoader(test_mat_ds, batch_size=batch_size)

# train_MF(epoch, lr, weight_decay, num_users, num_items, latent_dim, device)
train_AutoRec(epoch, lr, weight_decay, latent_dim, num_users, device, dropout)