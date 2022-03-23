import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
THRESHOLD = 1  # Threshold constant for the algorithm
DELTA = 0.001  # surroundings of the cluster delta constant

# gets the cluster center by its label
def get_cluster_center(cluster_label):
    index = np.where(model.subcluster_labels_ == cluster_label)[0]

    return model.subcluster_centers_[index]

# distance between two points
def distance(point, pred):
    return np.sqrt(np.sum(np.square(point - get_cluster_center(pred))))

# we get the points that are outliers
def outlier_filter(data, pred, threshold=THRESHOLD, delta=DELTA):
    result = []
    for i in range(len(data)):
        if distance(data[i], pred[i]) - threshold > delta:
            result.append(data[i])
    return result

# we get the points that are near the cluster's border
def boundary_filter(data, pred, threshold=THRESHOLD, delta=DELTA):
    result = []
    for i in range(len(data)):
        dist = distance(data[i], pred[i]) - threshold
        if dist <= delta and dist >= 0:
            result.append(data[i])
    return result


# generate random data
data, clusters = make_blobs(
    n_samples=1000, centers=5, cluster_std=0.65, random_state=0)

# make the model using Birch
model = Birch(branching_factor=50, n_clusters=5, threshold=THRESHOLD)
model.fit(data)
# get predictions
pred = model.predict(data)

# get outliers
outliers = outlier_filter(data, pred, THRESHOLD, DELTA)
# get boundary points
boundary = boundary_filter(data, pred, THRESHOLD, DELTA)

# calculate fuzzy evaluations
print('Fuzzy evaluations:')
membership = (len(data) - len(outliers) - len(boundary)) / len(data)
print(f'Membership: {membership:.3f}')
nonmembership = (len(outliers)) / len(data)
print(f'Non-membership: {nonmembership:.3f}')
unceirtanity = 1.0 - membership - nonmembership
print(f'Unceirtanity: {unceirtanity:.3f}')

# plot the data
plt.scatter(data[:, 0], data[:, 1], c=pred)
plt.show()
