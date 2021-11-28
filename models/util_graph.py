import logging
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig

logger = logging.getLogger()

def create_graph_nx(node_df: pd.DataFrame,
                    edge_df: pd.DataFrame,
                    node_id_col: str="node_id",
                    edge_src_col: str="source",
                    edge_dst_col: str="target",
                    node_label_col: str="label", # TODO
                    edge_label_col: str="label", # TODO
                    node_attr_cols: list = [],
                    edge_attr_cols: list=[]):

    """ Create a networkx graph object
    """
    G = nx.MultiGraph()
    G.add_nodes_from(node_df[node_id_col].values.flatten().tolist())
    G.add_edges_from(edge_df[[edge_src_col, edge_dst_col]].values)

    # Add node features
    nx.set_node_attributes(G, node_df.set_index(node_id_col)[node_attr_cols].to_dict())

    # Add edge features
    # nx.set_edge_attributes(G, edge_df.set_index([edge_src_col, edge_dst_col])[edge_attr_cols].transpose().to_dict())
    # Note that the line above is preferred way, but to_dict drops duplicates. Thus, we naively loop through the edge rows...
    counter = 0
    for ind, row in edge_df.set_index([edge_src_col, edge_dst_col])[edge_attr_cols].iterrows():
        nx.set_edge_attributes(G, {(ind[0], ind[1], counter): row[edge_attr_cols].to_dict()})
        counter += 1

    logger.info("********************* CREATING NETWORKX GRAPH  ********************************")
    logger.info(f'** NX summary: {nx.info(G)}')
    logger.info(f'** # Nodes: {G.number_of_nodes()}')
    logger.info(f'** # Edges: {G.number_of_edges()}')
    logger.info(f'** Is multigraph?: {G.is_multigraph()}')
    logger.info(f'** Number of subgraph: {nx.number_connected_components(G)}')

    Gcc = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=len, reverse=True)
    cc_sizes = [len(cc.nodes()) for cc in list(Gcc)]
    logger.info(f"** Sizes of subgraphs: {cc_sizes}")
    logger.info("*******************************************************************************")


    # TODO: If weights exist, add. This is out of scope for MVP
    # TODO: Add timestamps for temporal graph analysis
    return G

def create_graph_ig(dist_mat: list,
                    graph_type: str=ig.ADJ_UNDIRECTED):
    """ Create a iGraph object

    Arguments:
        dist_mat (list of lists): 2D distance matrix in form of list or numpy matrix
        graph_type (str): either one of 'directed', 'undirected', 'min', 'max', 'plus', 'upper', and 'lower'

    Returns:
        igraph.Graph: graph object in igraph package
    """
    G = ig.Graph.Adjacency(dist_mat, mode=graph_type)
    return G