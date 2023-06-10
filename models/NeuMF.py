import torch
import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(self, user_num, item_num, feature_dim, layer_num):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.u_gmf = nn.Embedding(self.user_num, self.feature_dim)
        self.i_gmf = nn.Embedding(self.item_num, self.feature_dim)
        self.u_mlp = nn.Embedding(self.user_num, self.feature_dim)
        self.i_mlp = nn.Embedding(self.item_num, self.feature_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(2 * self.feature_dim, self.feature_dim),
        #     nn.ReLU()
        # )
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(2*self.feature_dim, self.feature_dim))
        self.mlp.append(nn.ReLU())
        for i in range(self.layer_num-1):
            self.mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            self.mlp.append(nn.ReLU())
        self.pre_layer = nn.Linear(2 * self.feature_dim, 1, bias=False)

    def forward(self, user_id, item_id):
        gmf = self.u_gmf(user_id) * self.i_gmf(item_id)
        mlp = self.mlp(torch.cat([self.u_mlp(user_id), self.i_mlp(item_id)], dim=1))
        y_pre = self.pre_layer(torch.cat([gmf, mlp], dim=1))
        return y_pre
