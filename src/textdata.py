# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Loads the dialogue corpus, builds the vocabulary
"""
import sys
reload(sys)
sys.setdefaultencoding('iso-8859-1')
import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
import argparse
from collections import OrderedDict

from cornelldata import CornellData
import mxnet as mx

class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """

    availableCorpus = OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args

        # Path variables
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        self.samplesDir = os.path.join(self.args.rootDir, 'data/samples/')
        self.samplesName = self._constructName()

        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion

        self.loadCorpus(self.samplesDir)

        # Plot some stats:
        print('Loaded {}: {} words, {} QA'.format(self.args.corpus, len(self.word2id), len(self.trainingSamples)))

        if self.args.playDataset:
            self.playDataset()

    def _constructName(self):
        """Return the name of the dataset that the program should use with the current parameters.
        Computer from the base name, the given tag (self.args.datasetTag) and the sentence length
        """
        baseName = 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            baseName += '-' + self.args.datasetTag
        return '{}-{}.pkl'.format(baseName, self.args.maxLength)

    def makeLighter(self, ratioDataset):
        """Only keep a small fraction of the dataset, given by the ratio
        """
        #if not math.isclose(ratioDataset, 1.0):
        #    self.shuffle()  # Really ?
        #    print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]
            if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer
                sample = list(reversed(sample))
            if not self.args.test and self.args.autoEncode:  # Autoencode: use either the question or answer for both input and output
                k = random.randint(0, 1)
                sample = (sample[k], sample[k])
            batch.encoderSeqs.append(list(reversed(sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

            # Add padding & define weight
            batch.encoderSeqs[i]   = [self.padToken] * (self.args.maxLengthEnco  - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]  # Left padding for the input
            batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [self.padToken] * (self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        # Simple hack to reshape the batch
        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        # # Debug
        # self.printBatch(batch)  # Input inverted, padding should be correct
        # print(self.sequence2str(samples[0][0]))
        # print(self.sequence2str(samples[0][1]))  # Check we did not modified the original sample

        return batch

    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize())]

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches

    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.trainingSamples)

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)

    def loadCorpus(self, dirName):
        """Load/create the conversations data
        Args:
            dirName (str): The directory where to load/save the model
        """
        datasetExist = False
        if os.path.exists(os.path.join(dirName, self.samplesName)):
            datasetExist = True

        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')

            optionnal = ''
            if self.args.corpus == 'lightweight' and not self.args.datasetTag:
                raise ValueError('Use the --datasetTag to define the lightweight file to use.')
            else:
                optionnal = '/' + self.args.datasetTag  # HACK: Forward the filename

            # Corpus creation
            corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir + optionnal)
            self.createCorpus(corpusData.getConversations())

            # Saving
            print('Saving dataset...')
            self.saveDataset(dirName)  # Saving tf samples
        else:
            print('Loading dataset from {}...'.format(dirName))
            self.loadDataset(dirName)

        assert self.padToken == 0

    def saveDataset(self, dirName):
        """Save samples to file
        Args:
            dirName (str): The directory where to load/save the model
        """

        with open(os.path.join(dirName, self.samplesName), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'trainingSamples': self.trainingSamples
                }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, dirName):
        """Load samples from file
        Args:
            dirName (str): The directory where to load the model
        """
        with open(os.path.join(dirName, self.samplesName), 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.trainingSamples = data['trainingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words

    def createCorpus(self, conversations):
        """Extract all data from the given vocabulary
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId('<go>')  # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # Word dropped from vocabulary

        # Preprocessing data

        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)

        # The dataset will be saved in the same order it has been extracted

    def extractConversation(self, conversation):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """

        # Iterate over all the lines of the conversation
        for i in tqdm_wrap(range(len(conversation['lines']) - 1),  # We ignore the last line (no answer for it)
                           desc='Conversation', leave=False):
            inputLine  = conversation['lines'][i]
            targetLine = conversation['lines'][i+1]

            inputWords  = self.extractText(inputLine['text'])
            targetWords = self.extractText(targetLine['text'], True)

            if inputWords and targetWords:  # Filter wrong samples (if one of the list is empty)
                self.trainingSamples.append([inputWords, targetWords])

    def extractText(self, line, isTarget=False):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
            isTarget (bool): Define the question on the answer
        Return:
            list<int>: the list of the word ids of the sentence
        """
        words = []

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if not isTarget:
                i = len(sentencesToken)-1 - i

            tokens = nltk.word_tokenize(sentencesToken[i])

            # If the total length is not too big, we still can add one more sentence
            if len(words) + len(tokens) <= self.args.maxLength:
                tempWords = []
                for token in tokens:
                    tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

                if isTarget:
                    words = words + tempWords
                else:
                    words = tempWords + words
            else:
                break  # We reach the max length already

        return words

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # Get the id if the word already exist
        wordId = self.word2id.get(word, -1)

        # If not, we create a new entry
        if wordId == -1:
            if create:
                wordId = len(self.word2id)
                self.word2id[word] = wordId
                self.id2word[wordId] = word
            else:
                wordId = self.unknownToken

        return wordId

    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = [label]
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        a = [(n, x.shape) for n, x in zip(self.data_names, self.data)]
        return a

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class CornellDataIter(mx.io.DataIter):
    def __init__(self, textData, buckets, batch_size, init_states, forward_data_feed, data_name="data", label_name="label"):
        super(CornellDataIter, self).__init__()
        self.textData = textData
        self.vocab_size = textData.getVocabularySize()
        self.data_name = data_name
        self.label_name = label_name
        self.batch_size = batch_size
        #self.num_layers = num_layers
        buckets.sort()
        self.buckets = buckets
        self.forward_data_feed = forward_data_feed
        self.data = [[] for _ in buckets]
        self.default_bucket_key = max(buckets)

        self.batch_size = batch_size

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (self.batch_size, self.default_bucket_key)),('decoding_data',(self.batch_size,self.default_bucket_key + 2))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key + 2))]
        self.curr_idx = 0
        self.batches = self.textData.getBatches()

    def __iter__(self):
        return self

    def next(self):
        if self.curr_idx < len(self.batches) - 1: # the last batch is not fit batch_size, so not use.

            batch = self.batches[self.curr_idx]
            self.curr_idx += 1
            encodeMatrix = np.matrix(batch.encoderSeqs)
            encodeBatchMajor = encodeMatrix.transpose()
            decodeMatrix = np.matrix(batch.decoderSeqs)
            decodeBatchMajor = decodeMatrix.transpose()
            labelMatrix = np.matrix(batch.targetSeqs)
            labelBatchMajor = labelMatrix.transpose()

            init_state_names = [x[0] for x in self.init_states]
            batch_encoding_input = mx.nd.array(encodeBatchMajor)
            batch_decoding_input = mx.nd.array(decodeBatchMajor)
            label_all = mx.nd.array(labelBatchMajor)
            data_all = [batch_encoding_input, batch_decoding_input] + self.init_state_arrays
            data_names = ["data", "decoding_data"] + init_state_names
            label_names = ["softmax_label"]
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all, self.default_bucket_key)
            return data_batch
        else:
            raise StopIteration

    def reset(self):
        self.textData.shuffle()
        self.batches = self.textData.getBatches()
        self.curr_idx = 0


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

trainingArgs = parser.add_argument_group('Training options')
trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
trainingArgs.add_argument('--saveEvery', type=int, default=2000, help='nb of mini-batch step before creating a model checkpoint')
trainingArgs.add_argument('--batchSize', type=int, default=256, help='mini-batch size')
trainingArgs.add_argument('--learningRate', type=float, default=0.002, help='Learning rate')
trainingArgs.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (keep probabilities)')

if __name__ == "__main__":
    args = parser.parse_args()
    textData = TextData(args)
    print('Dataset created! Thanks for using this program')
    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2
    batches = textData.getBatches()
    print textData.printBatch(batches[0]), len(batches[0].encoderSeqs)
    init_c = [("encode_init_c", (args.batchSize, 2, 1024))] # Need to fix
    init_h = [("encode_init_h", (args.batchSize, 2, 1024))] # Need to fix
    init_states = init_c + init_h
    data = CornellDataIter(textData, [args.maxLength,], args.batchSize, init_states, True)
