import logging, os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

logger = logging.getLogger()

def train(X, n_cluster):
    # TODO: this algorithm can be driven by distance threshold rather than n_cluster. To explore in future

    logger.info("********************* RUNNING KMEANS CLUSTERING *******************************")

    # Model Training
    model = AgglomerativeClustering(n_clusters=n_cluster, compute_distances=True).fit(X)
    output = model.labels_

    # Performance measure
    perf_measure = {'avg_distance': sum(model.distances_) / model.n_clusters_}

    return model, output, perf_measure