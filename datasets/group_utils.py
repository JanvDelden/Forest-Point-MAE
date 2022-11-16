import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pytorch3d.ops as ops
import faiss

def process(ref, query, n_neighbors=32):
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(ref) 
    val = knn.kneighbors(query, return_distance=False)
    return val


class Group():
    def __init__(self, num_group, group_size, sampling_method):
        self.num_group = num_group
        self.group_size = group_size
        self.sampling_method = sampling_method

    def group(self, points):
        return sample_and_group(points, self.sampling_method, num_group=self.num_group, n_neighbors=self.group_size)

def sample_and_group(points, sampling_method="random", num_group=512, n_neighbors=32):
    '''
        input: N 3
        ---------------------------
        output: B G M 3
        center : B G 3
    '''
    assert points.shape[1] == 3
    N, _ = points.shape
    if sampling_method == "rand":
        idx = np.random.choice(N, num_group)
        center = points[idx]

    elif sampling_method == "fps":
        if len(points) > 8192:
            pointsc = points[::6].copy()
        else:
            pointsc = points.copy()
        center, _ = ops.sample_farthest_points(torch.from_numpy(pointsc).unsqueeze(0), K =num_group,random_start_point=True)
        center = center[0].numpy()
    elif sampling_method == "slice_fps":
        n_slices = 10
        points_copy = points[np.argsort(points[:, 2])].copy()
        points_copy = points_copy[:, :3]
        length = len(points_copy) - len(points_copy) % n_slices
        slices = points_copy[:length].reshape(n_slices, -1, 3)
        num_tokens = int(np.ceil(num_group / n_slices))
        center = ops.sample_farthest_points(torch.from_numpy(slices), K=num_tokens,random_start_point=True)[0]
        center = center.view(-1, 3)
        center = center.numpy()
        idx = np.random.choice(len(center), num_group, replace=False)
        center = center[idx]
    elif sampling_method == "kmeans":
        kmeans = faiss.Kmeans(d=3, k=num_group, niter=10, nredo=1, min_points_per_centroid=1)
        points = points.astype("float32", order="C")
        kmeans.train(points)
        center = kmeans.centroids
    elif sampling_method == "fpskmeans":
        if len(points) > 8192:
            pointsc = points[::6].copy()
        else:
            pointsc = points.copy()
        center, _ = ops.sample_farthest_points(torch.from_numpy(pointsc).unsqueeze(0), K =num_group,random_start_point=True)
        center = center[0].numpy()
        kmeans = faiss.Kmeans(d=3, k=num_group, niter=10, nredo=1, min_points_per_centroid=1)
        pointsc = pointsc.astype("float32", order="C")
        kmeans.train(pointsc, init_centroids=center)
        center = kmeans.centroids
    elif sampling_method == "kmeans_jitter":
        kmeans = faiss.Kmeans(d=3, k=num_group, niter=10, nredo=1, min_points_per_centroid=1)
        points = points.astype("float32", order="C")
        kmeans.train(points)
        center = kmeans.centroids
        center += np.random.normal(loc=0.0, scale=0.6, size=center.shape)
    else:
        print(sampling_method)
        raise NotImplementedError
    idx = process(points, center, n_neighbors) # G x neighbors
    neighborhood = points[idx]
    return neighborhood, center, idx


class Mask():
    def __init__(self, mask_ratio, num_group, group_size, mask_type):
        self.num_mask =  int(mask_ratio * num_group)
        self.num_group = num_group
        self.group_size = group_size
        self.mask_type = mask_type
        self.mask_template = np.hstack((np.zeros(self.num_group-self.num_mask), np.ones(self.num_mask))).astype(bool)


    def mask(self, neighborhood, center):
        '''
        center :G 3
        neighborhood G N 3
        --------------
        mask : G (bool)
        '''
        if self.mask_type == "points":
            np.random.shuffle(self.mask_template)
            filter_points = center[self.mask_template]
            filter = neighborhood.reshape(-1, 3) == filter_points.reshape(self.num_mask, 1, 3)
            filter = filter.reshape(self.num_mask, self.num_group, self.group_size, 3)
            filter = np.any(filter, axis=(2,3))
            # check in which patches the filter points are present, choose num mask patches to mask
            for j in range(self.num_group):
                candidate = np.any(filter[:j], axis=0)
                difference = np.sum(candidate) - self.num_mask
                if difference >= 0:
                    break
                if j == self.num_group - 1:
                    raise NotImplementedError
            # cap bool at mask ratio length
            if difference > 0:
                idx = np.random.choice(candidate.sum(), difference, replace=False)
                idx = np.nonzero(candidate)[0][idx]
                candidate[idx] = False
        elif self.mask_type == "rand":
            np.random.shuffle(self.mask_template)
            candidate = self.mask_template
        else:
            print(self.mask_type)
            raise NotImplementedError
        assert candidate.sum() == self.num_mask
        return candidate


if __name__ == "main":
    neighborhood, center, idx = sample_and_group(np.random.rand(8000, 3))
    masker = Mask(0.5, 512, 32, "points")
    candidate = masker.mask(neighborhood, center)

