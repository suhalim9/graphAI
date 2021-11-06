import logging, joblib, json, pickle
import pandas as pd

logger = logging.getLogger()

def save_sklearn_model(model, filename):
    joblib.dump(model, filename)
    logger.info(f"** Successfully saved {model} in {filename}")

def save_clustering_output(node_id_list, cluster_id_list, filename):
    output_df = pd.DataFrame([node_id_list, cluster_id_list]).transpose()
    output_df.columns = ['node_id', 'cluster_id']
    output_df.to_csv(filename)
    logger.info(f"** Successfully saved clustering output in {filename}")

def save_model_performance(perf_dict, filename):
    with open(filename, 'w') as fh:
        json.dump(perf_dict, fh)
    logger.info(f"** Successfully saved performance dictionary in {filename}")


def load_sklearn_model(filename):
    return joblib.load(filename)