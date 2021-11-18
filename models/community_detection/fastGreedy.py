import sys, logging, os, datetime, json
this_filepath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, '/'.join(this_filepath.split('/')[:-1]))

import util_data, util_graph, util_model
import networkx as nx
import igraph as ig


########################################################### 
# Arguments to get from system
# 
# Example running this file
# python models/community_detection/fastGreedy.py "{\"node_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/nodes.csv\", \"edge_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/edges_withTotalPriceUSD.csv\", \"node_id_col\": \"Id\", \"node_attr_cols\": [], \"edge_src_col\": \"SellerAddress\", \"edge_dst_col\": \"WinnerAddress\", \"edge_attr_cols\": [\"TotalPriceUSD\"], \"output_path_model\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model.pkl\", \"output_path_perf_measure\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/perf_measure.json\", \"output_path_graph_data\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/graph_data.json\", \"output_path_model_pred\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model_output.csv\", \"output_path_logfile\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/logfile.log\" }"
###########################################################

sysargs = json.loads(sys.argv[1])

### Data definition
node_filename = sysargs['node_filename']                    #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/nodes.csv'
edge_filename = sysargs['edge_filename']                    #/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/edges_withTotalPriceUSD.csv'
node_id_col = sysargs['node_id_col']                        #'Id'
node_attr_cols = sysargs['node_attr_cols'] or []            #[]
edge_src_col = sysargs['edge_src_col']                      #'SellerAddress'
edge_dst_col = sysargs['edge_dst_col']                      #'WinnerAddress'
edge_attr_cols = sysargs['edge_attr_cols']                  #['TotalPriceUSD']
output_path_model = sysargs['output_path_model']            #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model.pkl'
output_path_perf_measure = sysargs['output_path_perf_measure'] #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/perf_measure.json'
output_path_graph_data = sysargs['output_path_graph_data']  #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/graph_data.json'
output_path_model_pred = sysargs['output_path_model_pred']  #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model_output.csv'
output_path_logfile = sysargs['output_path_logfile']        #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/logfile.log'

### Model parameters
# n_cluster = sysargs['n_cluster'] #5


########################################################### 
# Set up logger
###########################################################
logger = logging.getLogger()
dt_now = datetime.datetime.now()
logFilename = os.path.join(output_path_logfile)

fhandler = logging.FileHandler(filename=logFilename, mode='a')
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.ERROR)


########################################################### 
# Step 1. Read data and define graph
###########################################################
logger.info("********************* RECEIVED ARGUMENTS *************************************")
logger.info(f"----- Graph Definition")
logger.info(f'** Node data file:           {node_filename}')
logger.info(f'** Node ID column:           {node_id_col}')
logger.info(f'** Node attribute column:    {node_attr_cols}')
logger.info(f'** Edge filename:            {edge_filename}')
logger.info(f'** Edge ID columns:          {edge_src_col} -> {edge_dst_col}')
logger.info(f'** Edge attribute columns:   {edge_attr_cols}')
logger.info(f"----- Output Paths")
logger.info(f"** Model:                    {output_path_model}")
logger.info(f"** Performance measure:      {output_path_perf_measure}")
logger.info(f"** Graph data (node_link)    {output_path_graph_data}")
logger.info(f"** Model prediction:         {output_path_model_pred}")
logger.info(f"** Log file:                 {output_path_logfile}")
logger.info("*******************************************************************************")


node_df = util_data.read_node_data(node_filename, node_id_col, node_attr_cols)
edge_df = util_data.read_edge_data(edge_filename, edge_src_col, edge_dst_col, edge_attr_cols)

G_nx = util_graph.create_graph_nx(node_df, edge_df, 
                                  node_id_col=node_id_col, 
                                  edge_src_col=edge_src_col, edge_dst_col=edge_dst_col,
                                  node_attr_cols=node_attr_cols, edge_attr_cols=edge_attr_cols)
G_ig = util_graph.create_graph_ig(nx.to_numpy_matrix(G_nx), graph_type='undirected')
G_ig.summary()

########################################################### 
# Step 2. Run a model
###########################################################
logger.info("*******************************************************************************")
node_id_list = list(G_nx.nodes())
logger.info("Building infomap model")
model = G_ig.community_fastgreedy().as_clustering()
logger.info("Successfully built infomap model")
pred = model.membership
perf_measure = {'modularity': model.modularity}
logger.info(f"Performance measure: {perf_measure}")
logger.info("*******************************************************************************")

########################################################### 
# Step 3. Saving files
###########################################################
logger.info(f'** Trained model: {model}')
logger.info(f'** Saving the model at {output_path_model}')
util_model.save_sklearn_model(model, output_path_model)    
logger.info(f'** Saving the clustering assignment at {output_path_model_pred}')
util_model.save_clustering_output(node_id_list, pred, output_path_model_pred)
logger.info(f"** Performance Metrics Dictionary: {perf_measure}")
logger.info(f"** Saving the performance metrics: {output_path_perf_measure} ")
util_model.save_model_performance(perf_measure, output_path_perf_measure)
logger.info(f"** Saving networkx node_link json (with modeling output): {output_path_graph_data}")
util_model.save_nx_node_link_json(G_nx, node_id_list, pred, output_path_graph_data)
logger.info("*******************************************************************************")
logger.info("********************************** FINISHED ***********************************")
logger.info("*******************************************************************************")
