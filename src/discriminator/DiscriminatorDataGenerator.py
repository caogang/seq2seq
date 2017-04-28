import random
import os
import sys
import copy
import pickle
sys.path.append('../')
sys.path.append('./')
from params import getArgs
from textdata import TextData
from seq2seq_model import Seq2SeqInferenceModelCornellData
import mxnet as mx
import data_utils


class Batch:
    def __init__(self):
        self.question = []
        self.answer = []
        # 0: machine, 1: human
        self.labels = []

class DiscriminatorData():

    def __init__(self, args, textData, pretrainedSeq2Seq, forceRegenerate = False):
        self.args = args
        self.textData = textData
        self.model = pretrainedSeq2Seq
        self.forceRegenerate = forceRegenerate

        self.samplesName = 'discriminator.pkl'
        samplesDir = os.path.join(args.rootDir, 'data/samples/')
        self.trainingSamples = []
        self.validationSamples = []
        self.loadData(samplesDir)
        pass

    def __createBatch(self, samples):
        """
        Generate a batch using padding and reverse
        :return: a Batch (batchSize, Batch)
        """
        batch = Batch()
        batchSize = len(samples)
        # print(samples)
        for i in range(batchSize):
            sample = samples[i]
            batch.question.append(list(reversed(sample[0])))
            batch.answer.append(list(reversed(sample[1])))
            batch.labels.append(sample[2])
            batch.question[i] = [self.textData.padToken] * (self.args.maxLengthEnco - len(batch.question[i])) + batch.question[i]  # Left padding for the input
            batch.answer[i] = [self.textData.padToken] * (self.args.maxLengthEnco - len(batch.answer[i])) + batch.answer[i]  # Left padding for the input

        return batch

    def getBatches(self, type='train'):
        """
        This function will shuffle first, and return batches
        :return:list<Batch>, guarantee every batch is batchSize
        """
        #self.shuffle()

        src_data = None
        if type == 'train':
            src_data = self.trainingSamples
        elif type == 'validation':
            src_data = self.validationSamples
        else:
            raise ValueError("parameter type '%s' is error, should be 'train' or 'validation'." % type)

        batches = []


        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            # remove the batch whose length is not equal to batchSize
            removeLast = False

            for i in range(0, len(src_data), self.args.batchSize):
                if i + self.args.batchSize >= len(src_data):
                    removeLast = True
                yield src_data[i:min(i + self.args.batchSize, len(src_data))], removeLast

        for samples, last in genNextSamples():
            if last:
                break
            batch = self.__createBatch(samples)
            batches.append(batch)
        return batches

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)
        random.shuffle(self.validationSamples)
        pass

    def __checkValidTrainingSamples(self):
        valid = True

        orgLen = len(self.trainingSamples)

        for sample in self.trainingSamples:
            if sample[0] is None or sample[1] is None:
                self.trainingSamples.remove(sample)
                valid = False

        if not valid:
            print('Origin discriminator training size : ' + str(orgLen))
            print('Clean discriminator training size : ' + str(len(self.trainingSamples)))

        return valid

    def __checkValidValidationSamples(self):
        valid = True

        orgLen = len(self.validationSamples)

        for sample in self.validationSamples:
            if sample[0] is None or sample[1] is None:
                self.validationSamples.remove(sample)
                valid = False

        if not valid:
            print('Origin discriminator validation size : ' + str(orgLen))
            print('Clean discriminator validation size : ' + str(len(self.validationSamples)))

        return valid

    def loadData(self, dirName):
        datasetExist = False
        print(os.path.join(dirName, self.samplesName))
        print('ForceRegenerate : ' + str(self.forceRegenerate))
        if os.path.exists(os.path.join(dirName, self.samplesName)):
            datasetExist = True

        if not datasetExist or self.forceRegenerate:  # First time we load the database: creating all files
            print('Discriminator Training samples not found. Creating dataset...')

            self.generateNegetiveSamples(self.textData, self.model)

            self.__checkValidTrainingSamples()
            self.__checkValidValidationSamples()

            # Saving
            print('Saving discriminator dataset...')
            print 'discriminator dataset training : %d QA' %len(self.trainingSamples)
            print 'discriminator dataset validation : %d QA' %len(self.validationSamples)
            with open(os.path.join(dirName, self.samplesName), 'wb') as handle:
                data = {
                    'trainingSamples': self.trainingSamples,
                    'validationSamples': self.validationSamples
                }
                pickle.dump(data, handle, -1)  # Using the highest protocol available
        else:
            print('Loading discriminator dataset from {}...'.format(dirName))
            with open(os.path.join(dirName, self.samplesName), 'rb') as handle:
                data = pickle.load(handle)
                self.trainingSamples = data['trainingSamples']
                self.validationSamples = data['validationSamples']
            print 'discriminator dataset training : %d QA' %len(self.trainingSamples)
            print 'discriminator dataset validation : %d QA' %len(self.validationSamples)

            valid = self.__checkValidTrainingSamples()
            valid = self.__checkValidValidationSamples() and valid

            if not valid:
                print('Resaving discriminator dataset...')
                with open(os.path.join(dirName, self.samplesName), 'wb') as handle:
                    data = {
                        'trainingSamples': self.trainingSamples,
                        'validationSamples': self.validationSamples
                    }
                    pickle.dump(data, handle, -1)  # Using the highest protocol available

        pass

    def generateNegetiveSamples(self, textData, inferenceModel):
        # [[list<id>, list<id>], ...] no padding and reverse
        positiveSamples = copy.deepcopy(textData.trainingSamples)
        tmpSamples = random.sample(positiveSamples, len(positiveSamples) // 2)
        negetiveSamples = []

        process = 0
        lenMax = len(tmpSamples)

        # add human label
        for qaPair in positiveSamples:
            # human label
            qaPair.append(1)

        for qaPair in tmpSamples:
            q = qaPair[0]
            s = inferenceModel.response(inferenceModel.forward_beam(q)[0].get_concat_sentence())
            s = s.rstrip(' <eos>')
            negetiveA = textData.sentence2id(s)
            pair = []
            pair.append(q)
            pair.append(negetiveA)
            # machine label
            pair.append(0)
            negetiveSamples.append(pair)

            s = inferenceModel.forward_sample(q)
            s = s.rstrip(' <pad>')
            s = s.rstrip(' <eos>')
            negetiveA = textData.sentence2id(s)
            pair = []
            pair.append(q)
            pair.append(negetiveA)
            # machine label
            pair.append(0)
            negetiveSamples.append(pair)
            process += 1
            print 'process: ' + str(process) + '/ max: ' + str(lenMax)

        # [[list < id >, list < id >, human_label], ...]
        samples = positiveSamples
        print len(samples)
        print len(negetiveSamples)
        # assert len(self.trainingSamples) == len(negetiveSamples)
        samples.extend(negetiveSamples)
        print len(samples)

        random.shuffle(samples)

        i = 0
        for p in samples:
            print i
            if i < 7:
                print 'add train'
                self.trainingSamples.append(p)
                i += 1
            else:
                print 'add valid'
                self.validationSamples.append(p)
                i += 1
            if i == 10:
                i = 0
        pass

    def getSampleSize(self):
        return len(self.trainingSamples)

if __name__ == "__main__":
    args = getArgs()
    originData = TextData(args)
    print originData.getSampleSize()
    print originData.getVocabularySize()
    #print data.trainingSamples[0]
    #print originData.sequence2str(data.trainingSamples[0][0])
    #print originData.sequence2str(data.trainingSamples[0][1])
    #print originData.sentence2id(originData.sequence2str(data.trainingSamples[0][0]))

    batch_size = 1
    buckets = data_utils.BUCKETS
    num_hidden = args.hiddenSize
    num_embed = args.embeddingSize

    momentum = 0.0
    num_layer = args.numLayers
    learning_rate = args.learningRate
    beam_size = 5  # 10

    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2

    if args.load is None:
        args.load = 50

    devs = mx.context.gpu(2)
    _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
    model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                                             originData, num_hidden, num_embed, num_layer,
                                             arg_params, beam_size,
                                             ctx=devs, dropout=0.)

    data = DiscriminatorData(args, originData, model, forceRegenerate=False)
    batches = data.getBatches()
    print batches[0].question
    print batches[0].answer
    print batches[0].labels
    pass
