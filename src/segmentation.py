#from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#from scipy.spatial.distance import cdist
import numpy as np

def cluster_points(points, n_clusters, min_cluster_size=1):

    if len(points) <= n_clusters:
        return [points]

    # compute the distance from each point to each other point
    #distances = cdist(points, points)

    # exclude distances between identical points
    #distances[distances == 0] = np.inf

    # find the distance from each point to its furthest neighbour, then get the
    # greatest of those distances among all points
    #max_min_distance = np.median(np.min(distances, 0)) * 4

    # if the computed distance is bad, then just return the input points
    #if max_min_distance == 0 or max_min_distance == np.inf:
        #return [points]

    # use the DBSCAN clustering algorithm to get point clusters
    #dbscan = DBSCAN(eps=max_min_distance, min_samples=min_cluster_size).fit(points)
    #labels = dbscan.labels_
    kmeans = KMeans(n_clusters=n_clusters).fit(points)
    labels = kmeans.labels_

    # initialize the clusters, ignoring the outlier cluster with index -1
    # since we assume there are no outliers, then the outlier cluster is empty
    clusters = [[] for label in range(len(set(labels)) - (1 if -1 in labels else 0))]

    # assign each point to its cluster
    for i in range(len(points)):
        if labels[i] >= 0:
            clusters[labels[i] - 1].append(points[i])

    return clusters
