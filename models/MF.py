from torch import nn


class MF(nn.Module):
    def __init__(self, num_user, num_item, latent_dim):
        """

        :param num_user:
        :param num_item:
        :param latent_dim: the dimension of latent space
        """
        super().__init__()
        self.user = nn.Embedding(num_user, latent_dim)
        self.item = nn.Embedding(num_item, latent_dim)
        self.user_bias = nn.Embedding(num_user, 1)
        self.item_bias = nn.Embedding(num_item, 1)

    def forward(self, user, item):
        out = (self.user(user) * self.item(item)).sum(axis=1) + self.user_bias(user).squeeze() + self.item_bias(
            item).squeeze()
        return out
