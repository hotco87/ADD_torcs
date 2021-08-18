import numpy as np

kk = np.load("test_buffer_action.npy")
print(np.shape(kk))

from sklearn.cluster import KMeans
print("kmeans_start")
kmeans = KMeans(n_clusters=750, init='k-means++',max_iter=500, tol=1e-04).fit(s_a.cpu())
labels = kmeans.labels_
center = kmeans.cluster_centers_

