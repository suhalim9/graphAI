import logging
import util_model

from stellargraph.mapper import (AdjacencyPowerGenerator)
from stellargraph.layer import WatchYourStep
from stellargraph.losses import graph_log_likelihood

from tensorflow.keras import Model, regularizers

logger = logging.getLogger()


def compute_watchYourStep_embedding(G_stellar, node_ID_list, epochs, walk_number, emb_dimension, attention_reg, optimizer_type, learning_rate, batch_size):
    """ This requires StellarGraph object. 
    Note that as of now, only Adam is allowed as an optimizer
    """
    logger.info("CREATING WATCH-YOUR-STEP EMBEDDING")
    logger.info("Setting up generator using StellarGraph package (Adjaceycy Power Generator)")
    generator = AdjacencyPowerGenerator(G_stellar, num_powers=10)
    wys = WatchYourStep(
        generator,
        num_walks=walk_number,
        embedding_dimension=emb_dimension,
        attention_regularizer=regularizers.l2(attention_reg),
    )
    x_in, x_out = wys.in_out_tensors()
    model = Model(inputs=x_in, outputs=x_out)
    optimizer = util_model.get_keras_optimizer(optimizer_type, learning_rate)
    model.compile(loss=graph_log_likelihood, optimizer=optimizer)
    train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=10)
    logger.info("Training the WYS embedding model")
    history = model.fit(
        train_gen, epochs=epochs, verbose=0, 
        steps_per_epoch=int(len(node_ID_list) // batch_size)
    )
    embeddings = wys.embeddings()
    logger.info("Successfully finished embedding part!")
    return embeddings    