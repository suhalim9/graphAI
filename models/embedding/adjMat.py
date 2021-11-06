import logging
import numpy as np
import networkx as nx
from networkx.convert_matrix import to_numpy_array

logger = logging.getLogger()

def compute_adj_mat_dense(G):
    """ This requires networkx graph object"""
    logger.info("CREATING ADJACENCY MATRIX")
    adjMat = to_numpy_array(G)
    logger.info(f"Adj Mat shape: {adjMat.shape}")
    return adjMat