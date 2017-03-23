import random
import os
from params import getArgs
from textdata import TextData

class DiscriminatorData():

    def __init__(self, args, textData, pretrainedSeq2Seq, forceRegenerate = False):
        self.textData = textData
        self.model = pretrainedSeq2Seq

        self.samplesDir = os.path.join(self.args.rootDir, 'data/samples/')
        self.trainingSamples = textData.trainingSamples
        pass

    def getBatches(self):
        self.shuffle()

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)
        pass


if __name__ == "__main__":
    args = getArgs()
    originData = TextData(args)
    data = DiscriminatorData(args, originData, None)
    print originData.getSampleSize()
    print originData.getVocabularySize()
    print data.trainingSamples