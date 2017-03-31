import mxnet as mx
import numpy as np

class SimpleDiscriminatorBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = [label]
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        a = [(n, x.shape) for n, x in zip(self.data_names, self.data)]
        return a

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DiscriminatorDataIter(mx.io.DataIter):
    def __init__(self, discriminatorData, batchSize, initStates, inputSeqLen, outputSeqLen):
        super(DiscriminatorDataIter, self).__init__()
        self.batchSize = batchSize
        self.discriminatorData = discriminatorData

        self.default_bucket_key = inputSeqLen

        self.initStates = initStates
        self.initStateArrays = [mx.nd.zeros(x[1]) for x in initStates]
        self.provide_data = [('inputData', (self.batchSize, inputSeqLen)), ('outputData',(self.batchSize, outputSeqLen))] + initStates
        self.provide_label = [('softmaxLabel', (self.batchSize, 2))]
        self.curr_idx = 0
        self.batches = self.discriminatorData.getBatches()

        print dict(self.provide_data+self.provide_label)

    def __iter__(self):
        return self

    def next(self):
        if self.curr_idx < len(self.batches): # the last batch is not fit batch_size, so not use.

            batch = self.batches[self.curr_idx]
            self.curr_idx += 1

            init_state_names = [x[0] for x in self.initStates]
            batch_input_seq = mx.nd.array(batch.question)
            batch_output_seq = mx.nd.array(batch.answer)
            label_all = mx.nd.array(batch.labels)
            data_all = [batch_input_seq, batch_output_seq] + self.init_state_arrays
            data_names = ["inputData", "outputData"] + init_state_names
            label_names = ["softmaxLabel"]
            data_batch = SimpleDiscriminatorBatch(data_names, data_all, label_names, label_all)
            return data_batch
        else:
            raise StopIteration

    def reset(self):
        self.batches = self.textData.getBatches()
        self.curr_idx = 0
