import logging, joblib, json, pickle, torch
import networkx as nx
import pandas as pd
import numpy as np
from tensorflow import keras

logger = logging.getLogger()

def save_torch_model(model, output_path_model):
    torch.save(model.state_dict(), output_path_model)
    logger.info(f"Successfully saved torch model {model} in {output_path_model}")


def get_torch_optimizer(model, optimizer_type, learning_rate):
    logger.info("*********************** GETTING OPTIMIZER ************")
    logger.info(model.parameters())
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    else:
        logger.info(optimizer_type, "is currently not supported!")
        return None
    return optimizer


def get_data_split_marker(train_ratio, val_ratio, test_ratio, node_count):
    logger.info("***************************** SPLITTING DATA *************")
    split_index = np.zeros((node_count), dtype=np.int)
    split_index[:round(node_count * train_ratio)] = 1
    split_index[round(node_count*train_ratio): round(node_count*(train_ratio+val_ratio))] = 2
    split_index[round(node_count*(train_ratio+val_ratio)):] = 3

    np.random.shuffle(split_index)
    logger.info("Node count in train:\t", sum(split_index == 1))
    logger.info("Node count in val:\t", sum(split_index == 2))
    logger.info("Node count in test:\t", sum(split_index == 3))
    logger.info("Total Sum Check:", sum(split_index == 1) + sum(split_index == 2) + sum(split_index == 3) == node_count)
    return torch.tensor(split_index == 1), torch.tensor(split_index == 2), torch.tensor(split_index == 3)

def get_keras_optimizer(optimizer_type, learning_rate=None, momentum=None, rho=None, epsilon=None):
    logger.info("Getting Keras optimizer")
    logger.info(f"Optimizer type: {optimizer_type}")
    logger.info(f"Provided learning_rate: {learning_rate}")
    logger.info(f"Provided momentum: {momentum}")
    logger.info(f"Provided rho: {rho}")
    logger.info(f"Provided epsilon: {epsilon}")

    learning_rate = learning_rate if learning_rate is not None else 0.001
    momentum = momentum if momentum is not None else 0.0
    rho = rho if rho is not None else 0.9
    epsilon = epsilon if epsilon is not None else 1e-7
    
    if optimizer_type == "Adam":
        logger.info(f"Using learning_rate: {learning_rate}")
        logger.info(f"Using epsilon: {epsilon}")
        return keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    elif optimizer_type == "SGD":
        logger.info(f"Using learning_rate: {learning_rate}")
        logger.info(f"Using momentum: {momentum}")
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_type == "RMSprop":
        logger.info(f"Using learning_rate: {learning_rate}")
        logger.info(f"Using momentum: {momentum}")
        logger.info(f"Using rho: {rho}")
        logger.info(f"Using epsilon: {epsilon}")
        return keras.optimizers.RMSprop(learning_rate=learning_rate, 
            momentum=momentum, rho=rho, epsilon=epsilon)
    elif optimizer_type == "Adagrad":
        logger.info(f"Using learning_rate: {learning_rate}")
        logger.info(f"Using epsilon: {epsilon}")
        return keras.optimizers.Adagrad(learning_rate=learning_rate,
            epsilon=epsilon)

def save_sklearn_model(model, filename):
    joblib.dump(model, filename)
    logger.info(f"** Successfully saved {model} in {filename}")

def save_nx_node_link_json(G_nx, node_id_list, pred, output_path_graph_data):
    # First save prediction output in the networkx object
    for i, node_id in enumerate(node_id_list):
        nx.set_node_attributes(G_nx, {node_id: {'pred': int(pred[i])}})
    # Saving the node link data in json format. This is the d3 format
    node_link_data = nx.readwrite.node_link_data(G_nx)
    with open(output_path_graph_data, 'w') as outfile:
        json.dump(node_link_data, outfile)
    logger.info(f"** Successfully saved networkx model (with prediction output saved at node features) in {output_path_graph_data}")


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