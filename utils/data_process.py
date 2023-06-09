import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from args import get_args

# get parameters
arg = get_args()


def read_data_ml(csv_path):
    """
    read ml100k.csv file
    :param csv_path: the path of csv file
    :return:
    data -> DataFrame
    num_users -> int
    num_items -> int
    """
    names = ['user', 'item', 'rating', 'time_']
    data = pd.read_csv(csv_path, sep='\t', names=names)
    num_users = data['user'].unique().shape[0]
    num_items = data['item'].unique().shape[0]
    return data, num_users, num_items


def split_data_ml(data, mode='random', test_ratio=0.1):
    """
    split the data to train set and test set
    :param data: a dataframe
    :param mode: random by default(split the data to train set(1-test_ratio) and test set(test_ratio)),
        or else, choose each user's latest interaction as test set
    :param test_ratio: default 0.1, i.e., choose 10% as test set
    :return:
    train_data -> dataframe
    test_data -> dataframe
    """
    if mode == 'random':
        mask = [False if x < test_ratio else True for x in np.random.uniform(0, 1, data.shape[0])]
        test_mask = [not x for x in mask]
        train_data = data[mask]
        test_data = data[test_mask]
    else:
        train_item, test_item, train_list = {}, {}, []
        for _, user, item, rating, time_ in data.itertuples():
            train_item.setdefault(user, []).append((user, item, rating, time_))
            # put the latest interaction of per user into the test_item
            if user not in test_item or test_item[user][-1] < time_:
                test_item[user] = (user, item, rating, time_)
        test_item_set = {v for v in test_item.values()}
        for v in train_item.values():
            for e in v:
                if e not in test_item_set:
                    train_list.append(e)
        train_data = pd.DataFrame(train_list, columns=names)
        test_data = pd.DataFrame(test_item_set, columns=names)
    return train_data, test_data


def load_and_split(csv_path, split_mode='random', test_ratio=0.1, implicit=True):
    """
    - put read data and split data together
    - we substract 1 for user id and item id,
        for movielens is one-based indexing, however in python it is zero-based indexing
    :param csv_path:
    :param split_mode: random or not, see `split_data_ml` for detail
    :param test_ratio:
    :param implicit: if True, the interaction of user and item will be 1, else will be the actual rating score.
    :return:
    train_data -> dataframe
    test_data -> dataframe
    """
    data, num_users, num_items = read_data_ml(csv_path)
    data['user'] = data['user'] - 1
    data['item'] = data['item'] - 1
    train_data, test_data = split_data_ml(data=data, mode=split_mode, test_ratio=test_ratio)
    if implicit:
        train_data = train_data.drop(columns=['rating', 'time_'])
        test_data = test_data.drop(columns=['rating', 'time_'])
    return train_data, test_data, num_users, num_items


def get_rating_matrix(data, num_user, num_item, implicit=True):
    """
    This function is write for AutoRec to prepare the rating matrix
    :param data: a dataframe
    :param num_user:
    :param num_item:
    :param implicit:
    :return:
    a rating matrix if implicit is False,
    or else, a set.
    """
    r_mat = {} if implicit else np.zeros((num_user, num_item))
    for l in data.itertuples():
        user, item = l[1], l[2]
        score = 1 if implicit else l[3]
        if implicit:
            r_mat.setdefault(user, []).append(item)
        else:
            r_mat[user, item] = score
    return r_mat


class Dataset_mat(Dataset):
    """
    - prepare date for AutoRec
    - if implicit is Ture, the data will be a set
        or else, the data will be a rating matrix, and it is np.ndarry type, and is_mat is True
    """
    def __init__(self, data):
        super().__init__()
        self.data = data
        if isinstance(self.data, np.ndarray):
            self.is_mat = True

    def __len__(self):
        if self.is_mat:
            return self.data.shape[1]
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.is_mat:
            return self.data[:, idx]
        else:
            return self.data[idx]

class Dataset_ml(Dataset):
    """
    prepare data for MF.
    """
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.num = data.shape[0]
    def __getitem__(self, idx):
        return self.data.iloc[idx].tolist()
    def __len__(self):
        return self.num