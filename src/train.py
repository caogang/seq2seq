import mxnet as mx
import numpy as np
import data_utils
from bucket_io import BucketSentenceIter
from lstm import seq2seq_lstm_unroll
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Global options
globalArgs = parser.add_argument_group('Global options')
globalArgs.add_argument('--test',
                                nargs='?',
                                default=None,
                                help='if present, launch the program try to answer all sentences from data/test/ with'
                                     ' the defined model(s), in interactive mode, the user can wrote his own sentences,'
                                     ' use daemon mode to integrate the chatbot in another program')
globalArgs.add_argument('--rootDir', type=str, default="../", help='folder where to look for the models and data')
globalArgs.add_argument('--playDataset', type=int, nargs='?', const=10, default=None,  help='if set, the program  will randomly play some samples(can be use conjointly with createDataset if this is the only action you want to perform)')
globalArgs.add_argument('--autoEncode', action='store_true', help='Randomly pick the question or the answer and use it both as input and output')
globalArgs.add_argument('--watsonMode', action='store_true', help='Inverse the questions and answer when training (the network try to guess the question)')

# Dataset options
datasetArgs = parser.add_argument_group('Dataset options')
datasetArgs.add_argument('--corpus', choices=TextData.corpusChoices(), default=TextData.corpusChoices()[0], help='corpus on which extract the dataset.')
datasetArgs.add_argument('--datasetTag', type=str, default='', help='add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions. Also used to define the file used for the lightweight format.')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\'
datasetArgs.add_argument('--ratioDataset', type=float, default=1.0, help='ratio of dataset used to avoid using the whole dataset')  # Not implemented, useless ?
datasetArgs.add_argument('--maxLength', type=int, default=10, help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')
datasetArgs.add_argument('--lightweightFile', type=str, default=None, help='file containing our lightweight-formatted corpus')

# Network options (Warning: if modifying something here, also make the change on save/loadParams() )
nnArgs = parser.add_argument_group('Network options', 'architecture related option')
nnArgs.add_argument('--hiddenSize', type=int, default=512, help='number of hidden units in each RNN cell')
nnArgs.add_argument('--numLayers', type=int, default=2, help='number of rnn layers')
nnArgs.add_argument('--embeddingSize', type=int, default=64, help='embedding size of the word representation')
nnArgs.add_argument('--initEmbeddings', action='store_true', help='if present, the program will initialize the embeddings with pre-trained word2vec vectors')
nnArgs.add_argument('--softmaxSamples', type=int, default=0, help='Number of samples in the sampled softmax loss function. A value of 0 deactivates sampled softmax')

trainingArgs = parser.add_argument_group('Training options')
trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
trainingArgs.add_argument('--saveEvery', type=int, default=2000, help='nb of mini-batch step before creating a model checkpoint')
trainingArgs.add_argument('--batchSize', type=int, default=256, help='mini-batch size')
trainingArgs.add_argument('--learningRate', type=float, default=0.002, help='Learning rate')
trainingArgs.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (keep probabilities)')

if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batchSize
    buckets = [args.maxLength]
    num_hidden = args.hiddenSize
    num_embed = args.embeddingSize
    num_layers = args.numLayers
    num_epoch = args.numEpochs
    learning_rate = args.learningRate
    momentum = 0.0
    clip_norm = 1.0

    textData = TextData(args)
    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2

    devs = mx.context.gpu(0)

    # Put needed training stage here
    training_stage = [1]

    # Stage 1, Episode 1
    # Forward seq2seq

    if 1 in training_stage:
        def sym_gen(seq_len):
            return seq2seq_lstm_unroll(seq_len, num_hidden, num_embed,
                    num_vocab=textData.getVocabularySize, num_layer = num_layers, dropout=args.dropout)
        forward_seq2seq_sym = sym_gen

        init_c = [("encode_init_c", (batch_size, num_layers, num_hidden))]
        init_h = [("encode_init_h", (batch_size, num_layers, num_hidden))]
        init_states = init_c + init_h

        Forward_data_feed = True
        forward_data_train = CornellDataIter(textData, [args.maxLength,], args.batchSize, init_states, True)
                                        #BucketSentenceIter(target_path, vocab,
                                        #buckets, batch_size,num_layers, init_states, Forward_data_feed)

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
                  epoch_end_callback=mx.callback.do_checkpoint("../snapshots/seq2seq_newdata", period = 1))
