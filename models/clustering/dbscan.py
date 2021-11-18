import logging
from sklearn import metrics
from sklearn.cluster import DBSCAN

logger = logging.getLogger()

def train(X, eps, min_samples):
    logger.info("********************* RUNNING DBSCAN CLUSTERING *******************************")

    # Model Training
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    pred = model.labels_

    # Performance measure
    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    n_noise = list(pred).count(-1)
    silhouette_score = metrics.silhouette_score(X, pred)
    perf_measure = {'n_cluster': n_clusters, 'n_noise': n_noise, 'silhouette_score': silhouette_score}
    return model, pred, perf_measure