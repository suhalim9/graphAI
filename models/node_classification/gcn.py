import sys, logging, os, datetime, json
this_filepath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, '/'.join(this_filepath.split('/')[:-1]))

import util_data, util_graph, util_model
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


########################################################### 
# Arguments to get from system
# 
# Example running this file
# python models/node_classification/gcn.py 
# "{\"node_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/nodes.csv\", 
#   \"edge_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/edges_withTotalPriceUSD.csv\", 
#   \"node_id_col\": \"Id\", \"node_label_col\": \"label\", \"node_attr_cols\": [], 
#   \"edge_src_col\": \"SellerAddress\", \"edge_dst_col\": \"WinnerAddress\", \"edge_attr_cols\": [\"TotalPriceUSD\"], 
#   \"output_path_model\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model.pkl\", 
#   \"output_path_perf_measure\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/perf_measure.json\", 
#   \"output_path_graph_data\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/graph_data.json\", 
#   \"output_path_model_pred\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model_output.csv\", 
#   \"output_path_logfile\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/logfile.log\",
#   \"split_ratio\": [0.6,0.2,0.2], \"aggregator\": \"mean\", \"h_feat_dim\":[16,4], \"activation_func\": \"relu\",
#   \"loss_func\": \"cross_entropy\", \"optimizer_type\": \"Adam\", \"learning_rate\": 0.01, \"train_iter\":100
# }"
#
# python models/node_classification/gcn.py "{\"node_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/nodes.csv\", \"edge_filename\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/edges_withTotalPriceUSD.csv\", \"node_id_col\": \"Id\", \"node_label_col\": \"label\", \"node_attr_cols\": [], \"edge_src_col\": \"SellerAddress\", \"edge_dst_col\": \"WinnerAddress\", \"edge_attr_cols\": [\"TotalPriceUSD\"], \"output_path_model\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model.pkl\", \"output_path_perf_measure\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/perf_measure.json\", \"output_path_graph_data\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/graph_data.json\", \"output_path_model_pred\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model_output.csv\", \"output_path_logfile\": \"/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/logfile.log\",\"split_ratio\": [0.6,0.2,0.2], \"aggregator\": \"mean\", \"h_feat_dim\":[16,4], \"activation_func\": \"relu\", \"loss_func\": \"cross_entropy\", \"optimizer_type\": \"Adam\", \"learning_rate\": 0.01, \"train_iter\":100}"
###########################################################
sysargs = json.loads(sys.argv[1])
### Data definition
node_filename = sysargs['node_filename']                    #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/nodes.csv'
edge_filename = sysargs['edge_filename']                    #/Users/suhalim9/Documents/InceptCube/GraphAI/git/data/edges_withTotalPriceUSD.csv'
node_id_col = sysargs['node_id_col']                        #'Id'
node_label_col = sysargs['node_label_col']                  #'label'
node_attr_cols = sysargs['node_attr_cols'] or []            #[]
edge_src_col = sysargs['edge_src_col']                      #'SellerAddress'
edge_dst_col = sysargs['edge_dst_col']                      #'WinnerAddress'
edge_attr_cols = sysargs['edge_attr_cols']                  #['TotalPriceUSD']
output_path_model = sysargs['output_path_model']            #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model.pkl'
output_path_perf_measure = sysargs['output_path_perf_measure'] #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/perf_measure.json'
output_path_graph_data = sysargs['output_path_graph_data']  #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/graph_data.json'
output_path_model_pred = sysargs['output_path_model_pred']  #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/model_output.csv'
output_path_logfile = sysargs['output_path_logfile']        #'/Users/suhalim9/Documents/InceptCube/GraphAI/git/output/logfile.log'
print("--------------------------")
### Model parameters
training_split_ratio = sysargs['split_ratio'][0]
validation_split_ratio = sysargs['split_ratio'][1]
testing_split_ratio = sysargs['split_ratio'][2]
assert sum(sysargs['split_ratio']) == 1

aggregator = sysargs['aggregator']
h_feat_dim = sysargs['h_feat_dim']
activation_func = sysargs['activation_func']
loss_func = sysargs['loss_func']
optimizer_type = sysargs['optimizer_type']
learning_rate = sysargs['learning_rate']
train_iter = sysargs['train_iter']

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
logger.info(f"** Node label column:        {node_label_col}")
logger.info(f'** Node attribute column:    {node_attr_cols}')
logger.info(f'** Edge filename:            {edge_filename}')
logger.info(f'** Edge ID columns:          {edge_src_col} -> {edge_dst_col}')
logger.info(f'** Edge attribute columns:   {edge_attr_cols}')
logger.info(f"----- Model Training Parameters")
logger.info(f"** Split Ratio:              {sysargs['split_ratio']}")
logger.info(f"** Aggregator:               {aggregator}")
logger.info(f"** Hidden Feature Dimension: {h_feat_dim}")
logger.info(f"** Activation Function:      {activation_func}")
logger.info(f"** Loss Function:            {loss_func}")
logger.info(f"** Optimizer Type:           {optimizer_type}")
logger.info(f"** Learning Rate:            {learning_rate}")
logger.info(f"** Train Iteration Number:   {train_iter}")
logger.info(f"----- Output Paths")
logger.info(f"** Model:                    {output_path_model}")
logger.info(f"** Performance measure:      {output_path_perf_measure}")
logger.info(f"** Graph data (node_link)    {output_path_graph_data}")
logger.info(f"** Model prediction:         {output_path_model_pred}")
logger.info(f"** Log file:                 {output_path_logfile}")
logger.info("*******************************************************************************")


node_df = util_data.read_node_data(node_filename, node_id_col, node_attr_cols)
edge_df = util_data.read_edge_data(edge_filename, edge_src_col, edge_dst_col, edge_attr_cols)

G_dgl = util_graph.create_graph_dgl(node_df, edge_df, 
                                node_id_col=node_id_col, node_label_col=node_label_col,
                                edge_src_col=edge_src_col, edge_dst_col=edge_dst_col,
                                node_attr_cols=node_attr_cols, edge_attr_cols=edge_attr_cols)

feat_dim = len(node_attr_cols)
num_classes = len(node_df[node_label_col].unique())
logger.info(f"Identified {feat_dim} dimension for the node attributes")
logger.info(f"Identified {num_classes} classes for the node classes")

########################################################### 
# Step 2. Run a model
###########################################################

# 2.1 Split the data
logger.info("Splitting the data")
node_count = len(node_df[node_id_col].unique())
train_marker, val_marker, test_marker = util_model.get_data_split_marker(training_split_ratio, validation_split_ratio, testing_split_ratio, len(node_f))
G_dgl.ndata['train_mask'] = train_marker
G_dgl.ndata['val_mask'] = val_marker
G_dgl.ndata['test_mask'] = test_marker
logger.info(G_dgl.ndata)
    
# 2.2 Define the model
logger.info("Defining GCN")
from dgl.nn import GraphConv
class GCN(nn.Module):
    def __init__(self, feat_dim, h_feat_dim, num_classes):
        super(GCN, self).__init__()
        
        self.num_layer = len(h_feat_dim)
        self.h_feat_dim = h_feat_dim
        self.num_classes = num_classes
        
        self.layers = []
        prev_dim = feat_dim
        for i in range(self.num_layer):
            self.layers.append(GraphConv(prev_dim, h_feat_dim[i]))
            prev_dim = h_feat_dim[i]
        self.layers.append(GraphConv(prev_dim, num_classes))
        
    def forward(self, g, in_feat):
        prev_feat = in_feat
        for i in len(self.num_layer-1):
            h = self.layers[i](g, prev_feat)
            h = F.relu(h)
            prev_feat = h
        h = self.layers[-1](g)

# Create the model with given dimensions
model = GCN(feat_dim, h_feat_dim, num_classes)
optimizer = util_model.get_torch_optimizer(model, optimizer_type, learning_rate) 
logger.info("GOT MODEL AND OPTIMIZER READY")    
    
best_train_acc = 0
best_val_acc = 0
best_test_acc = 0
    
features = G_dgl.ndata['feat']
labels = G_dgl.ndata['label']
train_mask = G_dgl.ndata['train_mask']
val_mask = G_dgl.ndata['val_mask']
test_mask = G_dgl.ndata['test_mask']
logger.info("------------ STARTING TRAINING -----------")
perf_measure = []
for e in range(train_iter):
    logits = model(G_dgl, features)   # forward
    pred = logits.argmax(1)       # compute prediction

    # TODO check the loss function type. Currently only cross entropy is supported
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    val_acc = (pred[train_mask] == labels[val_mask]).float().mean()
    test_acc = (pred[train_mask] == labels[test_mask]).float().mean()

    # Save the best validation accuracy
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        
    # Backward prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e%5 == 0:
        perf_measure.append({"epoch": e, "loss": loss, "val_acc": val_acc, "best_val_acc": best_val_acc, "test_acc": test_acc, "best_test_acc": best_test_acc})
        logger.info("In epoch {}, loss: {:.3f}, val acc: {:.3f}, test acc: {.3f} (best{:.3f})".format(
            e, loss, val_acc, best_val_acc, test_acc, best_test_acc))


########################################################### 
# Step 3. Saving files
###########################################################
logger.info(f'** Trained model: {model}')
logger.info(f'** Saving the model at {output_path_model}')
util_model.save_torch_model(model, output_path_model)    
logger.info(f'** Saving the clustering assignment at {output_path_model_pred}')
util_model.save_clustering_output(list(G_dgl.nodes()), pred, output_path_model_pred)
logger.info(f"** Performance Metrics Dictionary: {perf_measure}")
logger.info(f"** Saving the performance metrics: {output_path_perf_measure} ")
util_model.save_model_performance(perf_measure, output_path_perf_measure)
logger.info(f"** Saving networkx node_link json (with modeling output): {output_path_graph_data}")
# util_model.save_nx_node_link_json(G_nx, node_id_list, pred, output_path_graph_data)
logger.info("*******************************************************************************")
logger.info("********************************** FINISHED ***********************************")
logger.info("*******************************************************************************")
