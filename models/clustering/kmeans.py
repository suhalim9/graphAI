import logging, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

logger = logging.getLogger()

def train(X, n_cluster):
    logger.info("********************* RUNNING KMEANS CLUSTERING *******************************")

    # Model Training
    model = KMeans(n_clusters=n_cluster).fit(X)
    output = model.labels_

    # Performance measure
    inertia = model.inertia_
    distortion = sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
    perf_measure = {'inertia': inertia, 'distortion': distortion}

    return model, output, perf_measure