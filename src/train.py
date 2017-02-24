import mxnet as mx 
import numpy as np 
import data_utils 
from bucket_io import BucketSentenceIter
from lstm import seq2seq_lstm_unroll
import logging
from mutual_information_model import MutualInformationModel
from dialogue_simulation_reinforcement_learning import DialogueSimulationRLModel
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)




if __name__ == "__main__":
    batch_size = 80
    buckets = data_utils.BUCKETS
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

    devs = mx.context.gpu(2)

    # Put needed training stage here
    training_stage = [4]

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
                  epoch_end_callback=mx.callback.do_checkpoint("../snapshots/forward_seq2seq", period = 10))

    # Stage 1, Episode 2
    # Backward seq2seq

    if 2 in training_stage:
        def sym_gen(seq_len):
            return seq2seq_lstm_unroll(seq_len, num_hidden, num_embed, num_vocab=len(vocab), num_layer = num_layers, dropout=0.)
        backward_seq2seq_sym = sym_gen

        init_c = [("encode_init_c", (batch_size, num_layers, num_hidden))]
        init_h = [("encode_init_h", (batch_size, num_layers, num_hidden))]
        init_states = init_c + init_h

        Forward_data_feed = False
        backward_data_train = BucketSentenceIter(target_path, vocab,
                                        buckets, batch_size,num_layers, init_states, Forward_data_feed)

        optimizer = mx.optimizer.SGD(momentum = momentum,
                                     learning_rate = learning_rate,
                                     clip_gradient = clip_norm)

        model = mx.model.FeedForward(ctx = devs,
                                     symbol = backward_seq2seq_sym,
                                     num_epoch = num_epoch,
                                     learning_rate = learning_rate,
                                     optimizer = optimizer,
                                     momentum = momentum,
                                     wd = 0,
                                     initializer = mx.initializer.Uniform(scale=0.07))
        model.fit(X = backward_data_train,
                  eval_metric = "accuracy",
                  batch_end_callback=mx.callback.Speedometer(batch_size, 50),
                  epoch_end_callback=mx.callback.do_checkpoint("../snapshots/backward_seq2seq", period = 10))

    # Stage 2
    # Maximum Mutual Information Training

    if 3 in training_stage:
        learning_rate = 0.00003
        _, forward_arg_params, __ = mx.model.load_checkpoint("../snapshots/forward_seq2seq", 60)
        _, backward_arg_params, __ = mx.model.load_checkpoint("../snapshots/backward_seq2seq", 300)
        mutual_information_data_path = "../data/mmi_training_data.txt"
        seq_len = 30
        MMImodel = MutualInformationModel(seq_len, 1, learning_rate, vocab, rev_vocab, num_hidden, num_embed, num_layers,
                forward_arg_params, backward_arg_params, beam_size = 10, data_path = mutual_information_data_path, ctx=mx.gpu(0), dropout=0.)
        MMI_num_epoch = 10
        MMImodel.train(MMI_num_epoch)

    # Stage 3
    # Dialogue simulation reinforcement Learning

    if 4 in training_stage:
        seq_len = 40
        _, forward_arg_params, __ = mx.model.load_checkpoint("../snapshots/forward_seq2seq", 60)
        _, backward_arg_params, __ = mx.model.load_checkpoint("../snapshots/backward_seq2seq", 300)
        dialogue_simulation_data_path = "../data/mmi_training_data.txt"
        dialogue_simulation_model = DialogueSimulationRLModel(seq_len, 1, learning_rate, vocab, rev_vocab, num_hidden, num_embed, 
                num_layers, dialogue_simulation_data_path, forward_arg_params, backward_arg_params, beam_size = 5, ctx=mx.gpu(0), dropout=0.)
        dialogue_simulation_num_epoch = 10
        dialogue_simulation_model.train(dialogue_simulation_num_epoch)