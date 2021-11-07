import logging, joblib, json, pickle
import pandas as pd
import requests
import tempfile

logger = logging.getLogger()

def save_sklearn_model(model, filename):
    fd, path = tempfile.mkstemp()
    joblib.dump(model, path)
    requests.put(filename, open(path, "rb"))
    logger.info(f"** Successfully saved {model} in {filename}")

def save_clustering_output(node_id_list, cluster_id_list, filename):
    output_df = pd.DataFrame([node_id_list, cluster_id_list]).transpose()
    output_df.columns = ['node_id', 'cluster_id']
    fd, path = tempfile.mkstemp()
    output_df.to_csv(path)
    requests.put(filename, open(path, "rb"))
    logger.info(f"** Successfully saved clustering output in {filename}")

def save_model_performance(perf_dict, filename):
    json_obj = json.dumps(perf_dict)
    requests.put(filename, json_obj)
    logger.info(f"** Successfully saved performance dictionary in {filename}")


def load_sklearn_model(filename):
    return joblib.load(filename)
