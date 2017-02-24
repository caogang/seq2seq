import mxnet as mx 
import numpy as np 
import data_utils 
from seq2seq_autoencoder_bucket_io_test import BucketSentenceIter
from lstm import seq2seq_lstm_unroll
import logging
from mutual_information_model import MutualInformationModel
from dialogue_simulation_reinforcement_learning import DialogueSimulationRLModel
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

buckets = [5, 10]


if __name__ == "__main__":
    batch_size = 80
    buckets = [5, 10]
    num_hidden = 1024
    num_embed = 512
    num_layers = 4
    num_epoch = 300
    learning_rate = 0.003
    momentum = 0.0 
    clip_norm = 1.0

    data_path = "../data/openSubtitle_truncate_100k.txt"
    target_path = "../data/tokenized_data_openSubtitle_100k.txt"
    vocabulary_path = "../data/vocabulary_openSubtitle_100k.txt"
    max_vocabulary_size = 30000
    data_utils.create_vocabulary(vocabulary_path, data_path, max_vocabulary_size)
    data_utils.data_to_token_ids(data_path, target_path, vocabulary_path)

    vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)

    devs = mx.context.gpu(3)

    training_stage = [1]

    # Stage 1, Episode 1
    # Forward seq2seq

    if 1 in training_stage:
        def sym_gen(seq_len):
            return seq2seq_lstm_unroll(seq_len, num_hidden, num_embed, num_vocab=len(vocab), num_layer = num_layers, dropout=0.)
        forward_seq2seq_sym = sym_gen

        init_c = [("encode_init_c", (batch_size, num_layers, num_hidden))]
        init_h = [("encode_init_h", (batch_size, num_layers, num_hidden))]
        init_states = init_c + init_h

        Forward_data_feed = True
        forward_data_train = BucketSentenceIter(target_path, vocab,
                                        buckets, batch_size,num_layers, init_states, Forward_data_feed)

        optimizer = mx.optimizer.SGD(momentum = momentum,
                                     learning_rate = learning_rate,
                                     clip_gradient = clip_norm)

        model = mx.model.FeedForward(ctx = devs,
                                     symbol = forward_seq2seq_sym,
                                     num_epoch = num_epoch,
                                     learning_rate = learning_rate,
                                     optimizer = optimizer,
                                     momentum = momentum,
                                     wd = 0,
                                     initializer = mx.initializer.Uniform(scale=0.07))
        model.fit(X = forward_data_train,
                  eval_metric = "accuracy",
                  batch_end_callback=mx.callback.Speedometer(batch_size, 50),
                  epoch_end_callback=mx.callback.do_checkpoint("../snapshots/autoencoder_test_bucket", period = 10))