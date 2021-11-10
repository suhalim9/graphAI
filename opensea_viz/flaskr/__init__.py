import os
from flask.helpers import url_for
import pandas as pd
from flask import Flask, render_template, request

node_df = pd.read_csv("flaskr/static/opensea_cluster_output_oct29.csv")
for c in node_df.columns:
    if c != 'Id':
        node_df[c] = node_df[c].astype("int")
edge_df = pd.read_csv("flaskr/static/edges.csv")

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass


    @app.route('/opensea')
    def opensea():
        counts_by_cluster = {}

        cluster_IDs = [c for c in node_df.columns if c != 'Id']
        for cluster_ID in cluster_IDs:
            temp = node_df[['Id', cluster_ID]].groupby(cluster_ID).count().reset_index()
            temp.columns = ['cluster', 'count']
            counts_by_cluster[cluster_ID] = temp.to_dict('records')

        return render_template('network_viz.html', 
            colorBy='cluster_exp0', 
            clusterIDs = cluster_IDs,
            counts_by_cluster= counts_by_cluster)

    def get_connected_wallets(wallet_ID):
        source_wallets = edge_df.loc[edge_df['target'] == wallet_ID, 'source'].unique().tolist()
        target_wallets = edge_df.loc[edge_df['source'] == wallet_ID, 'target'].unique().tolist()
        connected_wallets = set(source_wallets + target_wallets)
        return connected_wallets
        
    @app.route('/opensea_detail')
    def opensea_detail():
        expID = request.args.get("expID")
        clusterID = int(request.args.get("clusterID"))
        wallets = node_df.loc[node_df[expID] == clusterID, 'Id'].unique()
        wallets_df = pd.DataFrame(wallets, columns = ['wallet_ID'])
        wallets_df = wallets_df.join(edge_df[edge_df['target'].isin(wallets)][['index', 'target']].groupby('target').count(), on='wallet_ID')
        wallets_df.columns = ['wallet_ID', 'beingSeller_count']
        wallets_df = wallets_df.join(edge_df[edge_df['source'].isin(wallets)][['index', 'source']].groupby('source').count(), on='wallet_ID')
        wallets_df.columns = ['wallet_ID', 'beingSeller_count', 'beingWinner_count']
        wallets_df = wallets_df.join(edge_df[edge_df['target'].isin(wallets)][['TotalPriceUSD', 'target']].groupby('target').mean(), on='wallet_ID')
        wallets_df.columns = ['wallet_ID', 'beingSeller_count', 'beingWinner_count', 'avg_purchasePriceUSD']
        wallets_df = wallets_df.join(edge_df[edge_df['source'].isin(wallets)][['TotalPriceUSD', 'source']].groupby('source').mean(), on='wallet_ID')
        wallets_df.columns = ['wallet_ID', 'beingSeller_count', 'beingWinner_count', 'avg_purchasePriceUSD', 'avg_sellPriceUSD']
        wallets_df['connected_wallets'] = wallets_df['wallet_ID'].apply(lambda wallet_ID: get_connected_wallets(wallet_ID))
        wallets_df['connected_wallets_count'] = wallets_df['connected_wallets'].apply(lambda wallets: len(wallets))

        cols = ['target', 'source', 'Timestamp', 'TotalPrice', 'TotalPriceUSD', 'TokenId', 'AssetId', 'TransactionId', 'AssetOwnerAddress', 'TransactionHash', 'eth_open_price']
        transactions_df = edge_df.loc[(edge_df['target'].isin(wallets))|edge_df['source'].isin(wallets), cols]
        trans_by_wallets = {}
        for w in wallets:
            trans_by_wallets[w] = edge_df.loc[(edge_df['target']==w)|(edge_df['source']==w), cols].to_html()

        return render_template('opensea_detail.html', 
            wallets_profile = wallets_df[['wallet_ID', 'beingSeller_count', 'beingWinner_count', 'avg_purchasePriceUSD', 'avg_sellPriceUSD', 'connected_wallets_count']].to_html(),
            wallets_count = len(wallets_df),
            transactions_all = transactions_df.to_html(),
            transactions_all_count = len(transactions_df),
            trans_by_wallets = trans_by_wallets
            )


    @app.route('/viz_test')
    def viz_test():
        return render_template('viz_test.html')

    @app.route('/opensea_v2')
    def opensea_v2():
        clusterID = int(request.args.get("clusterID"))
        return render_template("network_viz2.html", cluster=clusterID)
    return app
