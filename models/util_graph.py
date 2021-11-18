import logging, dgl, torch
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig

logger = logging.getLogger()

def create_graph_dgl(node_df: pd.DataFrame,
                     edge_df: pd.DataFrame,
                     node_id_col: str="node_id",
                     node_label_col: str="label",
                     edge_src_col: str="source",
                     edge_dst_col: str="target",
                     edge_label_col: str="label", # TODO
                     node_attr_cols: list = [],
                     edge_attr_cols: list=[]):
    logger.info("********************* CREATING DGL GRAPH  *************************************")
    node_features = node_df[node_attr_cols].values
    node_labels = node_df[node_label_col].tolist() if node_label_col in node_df.columns else None
    edge_features = edge_df[edge_attr_cols].values
    edge_labels = edge_df[edge_label_col].tolist() if edge_label_col in edge_df.columns else None
    edge_src = edge_df[edge_src_col].to_numpy()
    edge_dst = edge_df[edge_dst_col].to_numpy()
    g = dgl.graph((edge_src, edge_dst))
    g.ndata['feat'] = torch.tensor(node_features)
    g.ndata['label'] = torch.tensor(node_labels)
    g.edata['feat'] = torch.tensor(edge_features)

    logger.info(f"** Graph Summary:             {g}")
    logger.info(f"** Number of Nodes:           {g.number_of_nodes()}")
    logger.info(f"** Number of Edges:           {g.number_of_edges()}")
    # print("Node 0 has {} degree".format(g.in_degrees(0)))
    # print("Destinations from Node 0:", g.successors(0))
    # print("{} nodes presucceed Node 0".format(len(g.predecessors(0))))
    # print("Node 0 has {} in_edges and {} out_edges".format(len(g.in_edges(0)), len(g.out_edges(0))))
    logger.info(f"** Is this multigraph?        {g.is_multigraph}")
    # print("Does this graph has node 329?", g.has_nodes(329))
    # print("Is there an edge between 329 and 324?", g.has_edges_between(329, 324))
    return g

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
        nx.set_edge_attributes(G, {(ind[0], ind[1], str(counter)): row[edge_attr_cols].to_dict()})
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