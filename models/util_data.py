import logging
import pandas as pd
logger = logging.getLogger()


def read_node_data(data_filename: str, 
                    node_id_col: str, 
                    node_attr_cols: list=[]) -> pd.DataFrame:

    df = pd.read_csv(data_filename).dropna(subset=[node_id_col])
    logger.info("********************* READING NODE DATA ***************************************")
    logger.info(f'** Columns in node data: {df.columns.tolist()}')
    logger.info(f'** Count in the node data: {len(df)}')
    logger.info(f'** Count of unique node IDs: {len(df[node_id_col].unique())}')
    if node_attr_cols:
        logger.info(f'** Node attribute columns summary: ')
        logger.info(df[node_attr_cols].describe())
    logger.info("*******************************************************************************")

    return df[[node_id_col] + node_attr_cols]

def read_edge_data(data_filename: str, 
                    src_node_id_col: str, 
                    dst_node_id_col: str, 
                    edge_attr_cols: list=[]) -> pd.DataFrame:
    df = pd.read_csv(data_filename).dropna(subset=[src_node_id_col, dst_node_id_col]).reset_index()
    logger.info("********************* READING EDGE DATA ***************************************")
    logger.info(f'** Columns in edge data: {df.columns.tolist()}')
    logger.info(f'** Count in the edge data: {len(df)}')
    if edge_attr_cols:
        logger.info(f'** Edge attribute columns summary: ')
        logger.info(df[edge_attr_cols].describe())
    logger.info("*******************************************************************************")

    return df[[src_node_id_col, dst_node_id_col] + edge_attr_cols]
