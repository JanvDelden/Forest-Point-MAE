from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import numpy as np
import torch

# How many cpus should be used at maximum in parallel
CPU_NUMBER = 16

def knn_sklearn(batch_ref, batch_query, num_neighbors):
    """knn implementation on the cpu 

    Args:
        batch_ref (array like): batch x N x dim
        batch_query (array like)): _batch x num_querys x dim
        num_neighbors (int): num neighbors points are found

    Returns:
       array like: batch x num_querys x num_neighbors x 3 
    """
    knn = NearestNeighbors(n_neighbors=num_neighbors)

    # function to be parallelized
    def process(ref, query, knn):
        knn.fit(ref) 
        val = knn.kneighbors(query)
        return val

    neighbors = Parallel(n_jobs=CPU_NUMBER)(delayed(process)(ref, query, knn) for ref, query in zip(batch_ref, batch_query)) # list as return
    neighbors = np.stack([ind for dist, ind in neighbors])
    return neighbors

# class for compatibility to knn_cuda function
class KNN():
    def __init__(self, k, transpose_mode):
        # transpose mode is for compatibility
        self.num_neighbors = k

    def __call__(self, ref, query):
        ref, query = ref.cpu(), query.cpu()
        return None, torch.tensor(knn_sklearn(ref, query, self.num_neighbors)).cuda()