def hit_one_user(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k]) if val in test_matrix]
    positive_item_num = len(hits_k)
    return positive_item_num


def auc_one_user(rankedlist, test_matrix, k):
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist) if val in test_matrix]
    max_rank = len(rankedlist) - 1
    auc = 1.0 * (max_rank - hits_all[0][0]) / max_rank if len(hits_all) > 0 else 0
    return auc
