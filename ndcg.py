#--- Import Libraries ---#
import numpy as np
from imp_main import 

# top_k= 10
def dcg(relevances, rank=10):
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)
 
def ndcg(relevances, rank=10):
    # max-min
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.
    return dcg(relevances, rank) / best_dcg

# http://fastml.com/evaluating-recommender-systems/
if __name__ == '__main__':
    print(ndcg([3, 5, 8, 2, 5, 2, 4, 5, 9, 6], rank=10))