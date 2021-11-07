import sys, logging, os, datetime, json
this_filepath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, '/'.join(this_filepath.split('/')[:-1]))

# from models import util_data, util_graph
# from models.embedding import adjMat
# from models.clustering import kmeans

import util_data, util_graph
from embedding import adjMat
from clustering import kmeans


###########################################################
# Arguments to get from system
#
# Example running this file
# python graphAI/models/community_detection/adjMat_kmeans.py \
# "{\"node_filename\":\"/Users/sammy/Downloads/kay_nodes.csv\",\"edge_filename\":\"/Users/sammy/Downloads/edges_withTotalPriceUSD.csv\",\"node_id_col\":\"Id\",\"edge_src_col\":\"SellerAddress\",\"edge_dst_col\":\"WinnerAddress\",\"edge_attr_cols\":[\"TotalPriceUSD\"],\"output_paths\":{\"model\":\"https://graph-ai.s3.us-east-1.amazonaws.com/model3.pkl?x-amz-acl=public-read-write\\u0026X-Amz-Algorithm=AWS4-HMAC-SHA256\\u0026X-Amz-Credential=AKIAQOSIMQS3N2X2575P%2F20211107%2Fus-east-1%2Fs3%2Faws4_request\\u0026X-Amz-Date=20211107T150538Z\\u0026X-Amz-Expires=900\\u0026X-Amz-SignedHeaders=host\\u0026X-Amz-Signature=bac59b52cf0d647f0e145ecd36ac81b7f8bbafc7e6d305a07050353dc0da762a\",\"output\":\"https://graph-ai.s3.us-east-1.amazonaws.com/output.csv?x-amz-acl=public-read-write\\u0026X-Amz-Algorithm=AWS4-HMAC-SHA256\\u0026X-Amz-Credential=AKIAQOSIMQS3N2X2575P%2F20211107%2Fus-east-1%2Fs3%2Faws4_request\\u0026X-Amz-Date=20211107T150715Z\\u0026X-Amz-Expires=900\\u0026X-Amz-SignedHeaders=host\\u0026X-Amz-Signature=2dc5805eefd0b4383f717c812874f7dd9fa304e5dbe5f435bda92d3656b61d75\",\"perf_measure\":\"https://graph-ai.s3.us-east-1.amazonaws.com/perf_measure.json?x-amz-acl=public-read-write\\u0026X-Amz-Algorithm=AWS4-HMAC-SHA256\\u0026X-Amz-Credential=AKIAQOSIMQS3N2X2575P%2F20211107%2Fus-east-1%2Fs3%2Faws4_request\\u0026X-Amz-Date=20211107T150815Z\\u0026X-Amz-Expires=900\\u0026X-Amz-SignedHeaders=host\\u0026X-Amz-Signature=7e268565901a873f936dc5302d26cb3847d7ce4a7fefc61aebe25ead18e12a53\"},\"node_attr_cols\":[],\"n_custer\":5,\"n_cluster\":5}"
###########################################################

sysargs = json.loads(sys.argv[1])

### Data definition
node_filename = sysargs['node_filename']    #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/nodes.csv'
edge_filename = sysargs['edge_filename']    #/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/edges_withTotalPriceUSD.csv'
node_id_col = sysargs['node_id_col']        #'Id'
node_attr_cols = sysargs['node_attr_cols'] or []  #[]
edge_src_col = sysargs['edge_src_col']      #'SellerAddress'
edge_dst_col = sysargs['edge_dst_col']      #'WinnerAddress'
edge_attr_cols = sysargs['edge_attr_cols']  #['TotalPriceUSD']
output_paths = sysargs['output_paths']        #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/'

### Model parameters
n_cluster = sysargs['n_cluster'] #5


###########################################################
# Set up logger
###########################################################
logger = logging.getLogger()
# dt_now = datetime.datetime.now()
# logFilename = os.path.join(output_path, 'log_{}.log'.format(dt_now.strftime("%Y-%m-%d_%H-%M-%S")))
#
# fhandler = logging.FileHandler(filename=logFilename, mode='a')
# formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
# fhandler.setFormatter(formatter)
# logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.ERROR)


###########################################################
# Step 1. Read data and define graph
###########################################################
logger.info("********************* RECEIVED PARAMETERS *************************************")
logger.info(f'** Node data file:           {node_filename}')
logger.info(f'** Node ID column:           {node_id_col}')
logger.info(f'** Node attribute column:    {node_attr_cols}')
logger.info(f'** Edge filename:            {edge_filename}')
logger.info(f'** Edge ID columns:          {edge_src_col} -> {edge_dst_col}')
logger.info(f'** Edge attribute columns:   {edge_attr_cols}')
logger.info("*******************************************************************************")

node_df = util_data.read_node_data(node_filename, node_id_col, node_attr_cols)
edge_df = util_data.read_edge_data(edge_filename, edge_src_col, edge_dst_col, edge_attr_cols)

G_nx = util_graph.create_graph_nx(node_df, edge_df,
                                  node_id_col=node_id_col,
                                  edge_src_col=edge_src_col, edge_dst_col=edge_dst_col,
                                  node_attr_cols=node_attr_cols, edge_attr_cols=edge_attr_cols)

###########################################################
# Step 2. Run a model
###########################################################
node_id_list = list(G_nx.nodes())
adjMat = adjMat.compute_adj_mat_dense(G_nx)
kmeans.run(adjMat, node_id_list, n_cluster, output_paths)
logger.info("*******************************************************************************")
logger.info("********************************** FINISHED ***********************************")
logger.info("*******************************************************************************")
