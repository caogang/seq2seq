import mxnet as mx
import numpy as np


class DiscriminatorDataProvider(mx.io.DataIter):
    def __init__(self, textData):
        super(DiscriminatorDataProvider, self).__init__()
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