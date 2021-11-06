import logging, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import util_model

logger = logging.getLogger()

def run(X, node_id_list, n_cluster, output_paths):
    logger.info("********************* RUNNING KMEANS CLUSTERING *******************************")

    # Model Training
    model = KMeans(n_clusters=n_cluster)
    output = model.fit_predict(X)
    model_filename = output_paths['model']
    output_filename = output_paths['output']
    perf_filename = output_paths['perf_measure']

    # Saving model and clustering output
    logger.info(f'** Trained model: {model}')
    logger.info(f'** Saving the model at {model_filename}')
    util_model.save_sklearn_model(model, model_filename)
    logger.info(f'** Saving the clustering assignment at {output_filename}')
    util_model.save_clustering_output(node_id_list, output, output_filename)

    # Evaluating and saving evaluation
    inertia = model.inertia_
    distortion = sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
    perf_measure_dict = {'inertia': inertia, 'distortion': distortion}
    logger.info(f"** Performance Metrics Dictionary: {perf_measure_dict}")
    logger.info(f"** Saving the performance metrics: {perf_filename} ")
    util_model.save_model_performance(perf_measure_dict, perf_filename)
    logger.info("*******************************************************************************")
