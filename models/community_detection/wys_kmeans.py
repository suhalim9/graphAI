import sys, logging, os, datetime, json
from stellargraph import StellarGraph
this_filepath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, '/'.join(this_filepath.split('/')[:-1]))

import util_data, util_graph, util_model
from embedding import watchYourStep
from clustering import kmeans


########################################################### 
# Arguments to get from system
# 
# Example running this file (Make this one line)
# python models/community_detection/wys_kmeans.py "{\"node_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/nodes.csv\", \"edge_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/edges_withTotalPriceUSD.csv\", \"node_id_col\": \"Id\", \"node_attr_cols\": [], \"edge_src_col\": \"SellerAddress\", \"edge_dst_col\": \"WinnerAddress\", \"edge_attr_cols\": [\"TotalPriceUSD\"], \"output_path_model\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model.pkl\", \"output_path_perf_measure\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/perf_measure.json\", \"output_path_graph_data\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/graph_data.json\", \"output_path_model_pred\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model_output.csv\", \"output_path_logfile\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/logfile.log\", \"walk_number\":100, \"attention_reg\":0.5, \"batch_size\": 128, \"epochs\": 5, \"emb_size\": 128, \"optimizer_type\": \"Adam\", \"learning_rate\": 0.001, \"n_cluster\":5}"
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
# For WYS
walk_number = sysargs['walk_number']                        # 100
attention_reg = sysargs['attention_reg']                    # 0.5
emb_size = sysargs['emb_size']                              # 128
optimizer_type = sysargs['optimizer_type']                  # Adam
learning_rate = sysargs['learning_rate']                    # 1e-3
batch_size = sysargs['batch_size']                          # 128
epochs = sysargs['epochs']                                  # 5

# For KMeans
n_cluster = sysargs['n_cluster'] #5


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
logger.info(f"----- Model Parameters for WYS Embedding")
logger.info(f"** walk_number:                 {walk_number}")
logger.info(f"** attention_reg:               {attention_reg}")
logger.info(f"** batch_size:                  {batch_size}")
logger.info(f"** epochs:                      {epochs}")
logger.info(f"** emb_size:                    {emb_size}")
logger.info(f"** optimizer_type:              {optimizer_type}")
logger.info(f"** learning_rate:               {learning_rate}")
logger.info(f"----- Model Parameters for KMeans Clustering")
logger.info(f"** n_cluster:                   {n_cluster}")
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
G_stellar = StellarGraph.from_networkx(G_nx)

########################################################### 
# Step 2. Run a model
###########################################################
node_id_list = list(G_nx.nodes())
wys = watchYourStep.compute_watchYourStep_embedding(G_stellar, node_id_list, epochs, walk_number, emb_size, attention_reg, optimizer_type, learning_rate, batch_size)
model, pred, perf_measure = kmeans.train(wys, n_cluster)

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
