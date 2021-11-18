import logging
import util_model

from stellargraph.data import (BiasedRandomWalk, UnsupervisedSampler)
from stellargraph.mapper import (Node2VecLinkGenerator, Node2VecNodeGenerator)
from stellargraph.layer import Node2Vec, link_classification
from tensorflow import keras

logger = logging.getLogger()

def compute_node2vec_embedding(G_stellar, node_ID_list, walk_number, walk_length, p, q, batch_size, epochs, emb_size, optimizer_type, learning_rate):
    """ This requires StellagGraph object"""

    logger.info("CREATING NODE2VEC EMBEDDING")
    logger.info("Setting up walker and sampler using StellarGraph package")
    walker = BiasedRandomWalk(G_stellar, n=walk_number, length=walk_length, p=p, q=q)
    unsupervised_samples = UnsupervisedSampler(G_stellar, nodes=node_ID_list, walker=walker)
    generator = Node2VecLinkGenerator(G_stellar, batch_size)
    node2vec = Node2Vec(emb_size, generator=generator)
    x_inp, x_out = node2vec.in_out_tensors()

    logger.info("Training link_classification model to learn node2vec embeddings")  
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)

    optimizer = util_model.get_keras_optimizer(optimizer_type, learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    history = model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    logger.info("Finished link prediction training. Now getting embeddings")
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = Node2VecNodeGenerator(G_stellar, batch_size).flow(node_ID_list)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1) # this is using 4 threads (or workers)
    logger.info("Finished node embedding process!")
    logger.info(f"Embedding size: {node_embeddings.shape}")
    return node_embeddings