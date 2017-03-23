import random
import os
import sys
import copy
import pickle
import data_utils
sys.path.append('../')
sys.path.append('./')
from params import getArgs
from textdata import TextData
from seq2seq_model import Seq2SeqInferenceModelCornellData
import mxnet as mx


class DiscriminatorData():

    def __init__(self, args, textData, pretrainedSeq2Seq, forceRegenerate = False):
        self.textData = textData
        self.model = pretrainedSeq2Seq

        self.sampleName = 'discriminator.pkl'
        samplesDir = os.path.join(args.rootDir, 'data/samples/')
        self.loadData(samplesDir)
        pass

    def getBatches(self):
        self.shuffle()

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)
        pass

    def loadData(self, dirName):
        datasetExist = False
        if os.path.exists(os.path.join(dirName, self.samplesName)):
            datasetExist = True

        if not datasetExist:  # First time we load the database: creating all files
            print('Discriminator Training samples not found. Creating dataset...')

            self.generateNegetiveSamples(self.textData, self.model)

            # Saving
            print('Saving dataset...')
            with open(os.path.join(dirName, self.samplesName), 'wb') as handle:
                data = {
                    'trainingSamples': self.trainingSamples
                }
                pickle.dump(data, handle, -1)  # Using the highest protocol available
        else:
            print('Loading dataset from {}...'.format(dirName))
            with open(os.path.join(dirName, self.samplesName), 'rb') as handle:
                data = pickle.load(handle)
                self.trainingSamples = data['trainingSamples']
        pass

    def generateNegetiveSamples(self, textData, inferenceModel):
        # [[list<id>, list<id>], ...] no padding and reverse
        positiveSamples = copy.deepcopy(textData.trainingSamples)
        negetiveSamples = []

        for qaPair in positiveSamples:
            q = qaPair[0]
            a = qaPair[1]
            str = inferenceModel.response(inferenceModel.forward_beam(q)[0].get_concat_sentence())
            print textData.sentence2str(q)
            print textData.sentence2str(a)
            print str
            break

if __name__ == "__main__":
    args = getArgs()
    originData = TextData(args)
    data = DiscriminatorData(args, originData, None)
    print originData.getSampleSize()
    print originData.getVocabularySize()
    print data.trainingSamples[0]
    print originData.sequence2str(data.trainingSamples[0][0])
    print originData.sequence2str(data.trainingSamples[0][1])
    print originData.sentence2id(originData.sequence2str(data.trainingSamples[0][0]))

    batch_size = 1
    buckets = data_utils.BUCKETS
    num_hidden = args.hiddenSize
    num_embed = args.embeddingSize

    momentum = 0.0
    num_layer = args.numLayers
    learning_rate = args.learningRate
    beam_size = 10  # 10

    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2

    if args.load is None:
        args.load = 50

    devs = mx.context.gpu(0)
    _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
    model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                                             originData, num_hidden, num_embed, num_layer,
                                             arg_params, beam_size,
                                             ctx=devs, dropout=0.)

    data.generateNegetiveSamples(originData, model)
    pass
