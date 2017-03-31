import argparse
from textdata import TextData

def getArgs():
    parser = argparse.ArgumentParser()

    # Global options
    globalArgs = parser.add_argument_group('Global options')
    globalArgs.add_argument('--load', type=int, default=None,  help='if set, will load the corresponded checkpoint to do test(only valid for predict*.py and dump*.py)')
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

    # Discriminator options
    disArgs = parser.add_argument_group('Discriminator options')
    disArgs.add_argument('--inputLayerNums', type=int, default=1)
    disArgs.add_argument('--inputHiddenNums', type=int, default=512)
    disArgs.add_argument('--outputLayerNums', type=int, default=1)
    disArgs.add_argument('--outputHiddenNums', type=int, default=512)
    disArgs.add_argument('--contentLayerNums', type=int, default=1)
    disArgs.add_argument('--contentHiddenNums', type=int, default=512)


    trainingArgs = parser.add_argument_group('Training options')
    trainingArgs.add_argument('--numEpochs', type=int, default=5000, help='maximum number of epochs to run')
    trainingArgs.add_argument('--saveEvery', type=int, default=2000, help='nb of mini-batch step before creating a model checkpoint')
    trainingArgs.add_argument('--batchSize', type=int, default=256, help='mini-batch size')
    trainingArgs.add_argument('--learningRate', type=float, default=0.002, help='Learning rate')
    trainingArgs.add_argument('--dropout', type=float, default=0., help='Dropout rate (keep probabilities)')

    return parser.parse_args()
