import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
import data_utils
import logging




def read_data(path, data, forward_data_feed):
    count = 0
    with open(path, 'r') as read_file:
        first_line = read_file.readline()
        second_line = read_file.readline()
        third_line = read_file.readline()
        while third_line:
            if second_line == third_line:
                first_line = third_line
                second_line = read_file.readline()
                third_line = read_file.readline()
                continue
            elif second_line == data_utils.file_termimal:
                first_line = third_line
                second_line = read_file.readline()
                third_line = read_file.readline()
            elif third_line == data_utils.file_termimal:
                first_line = read_file.readline()
                second_line = read_file.readline()
                third_line = read_file.readline()
                continue
            else:
                first_line_ids = [int(x) for x in first_line.rstrip().split()]
                second_line_ids = [int(x) for x in second_line.rstrip().split()]
                third_line_ids = [int(x) for x in third_line.rstrip().split()]
                if forward_data_feed:
                    #### NEED TO CHECK THE DATA FEED, REMEMBER GO_ID, AVOID EOS_ID MULTIPLE TIMES
                    first_line_ids.extend(second_line_ids)
                    source_ids = first_line_ids
                    third_line_ids.append(data_utils.EOS_ID)
                    target_ids = list(third_line_ids)
                else:
                    source_ids = list(third_line_ids)
                    second_line_ids.append(data_utils.EOS_ID)
                    target_ids = list(second_line_ids)

            for bucket_id, seq_size in enumerate(data_utils.BUCKETS):
                if len(source_ids) < seq_size and len(target_ids) < seq_size:
                    data[bucket_id].append([source_ids, target_ids])
                    break

            first_line = second_line
            second_line = third_line
            third_line = read_file.readline()

            count += 1
            if count % 100000 == 0:
                logging.debug("Reading data line %d" % count)

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = [label]
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        self.pad = -1
        self.index = None

    @property
    def provide_data(self):
        a = [(n, x.shape) for n, x in zip(self.data_names, self.data)]
        return a

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, vocab, buckets, batch_size, num_layers, init_states, forward_data_feed, data_name="data", label_name="label"):
        super(BucketSentenceIter, self).__init__()
        self.vocab_size = len(vocab)
        self.data_name = data_name 
        self.label_name = label_name 
        self.batch_size = batch_size
        self.num_layers = num_layers
        buckets.sort() 
        self.buckets = buckets 
        self.forward_data_feed = forward_data_feed
        self.data = [[] for _ in buckets]
        self.default_bucket_key = max(buckets)
        read_data(path, self.data, self.forward_data_feed)

        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size) )

        self.batch_size = batch_size
        self.make_data_iter_plan() 

        self.init_states = init_states 
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (self.batch_size, self.default_bucket_key)),('decoding_data',(self.batch_size,self.default_bucket_key))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(len(self.data[i]) / self.batch_size) 
            self.data[i] = self.data[i][:int(bucket_n_batches[i] * self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int) + i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all 
        self.bucket_curr_idx = [0 for x in self.data]

        self.encoding_data_buffer = []
        self.decoding_data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            encoding_data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            decoding_data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            self.encoding_data_buffer.append(encoding_data)
            self.decoding_data_buffer.append(decoding_data)
            self.label_buffer.append(label)

    def padding(self, data_batch, bucket_size):
        encoding_input, decoding_input = [], [] 
        seq_len = bucket_size
        for encoding_sentence, decoding_sentence in data_batch:
            encoding_pad = [-1] * (seq_len - len(encoding_sentence))
            encoding_input.append(list(reversed(encoding_sentence + encoding_pad)))

            decoding_pad_size = seq_len - len(decoding_sentence) - 1
            decoding_input.append([data_utils.GO_ID] + decoding_sentence + [-1] * decoding_pad_size)

        batch_encoding_input = [None] * self.batch_size
        batch_decoding_input = [None] * self.batch_size
        for i in xrange(self.batch_size):
            batch_encoding_input[i] = np.array(encoding_input[i])
            batch_decoding_input[i] = np.array(decoding_input[i])

        return batch_encoding_input, batch_decoding_input

    def __iter__(self):
        for i_bucket in self.bucket_plan:
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx : i_idx + self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            init_state_names = [x[0] for x in self.init_states]
            data_batch_tmp = [self.data[i_bucket][i] for i in idx]
            bucket_size = self.buckets[i_bucket]
            batch_encoding_input, batch_decoding_input = self.padding(data_batch_tmp, bucket_size) 
            batch_encoding_input = np.array(batch_encoding_input)
            batch_decoding_input = np.array(batch_decoding_input)
            label = self.label_buffer[i_bucket]
            label[:,:-1] = batch_decoding_input[:,1:]
            label[:, -1] = -1
            # for rowidx in range(label.shape[0]):
            #     for columnidx in range(label.shape[1]):
            #         if label[rowidx, columnidx] == 0:
            #             label[rowidx,columnidx] = -1

            batch_encoding_input = mx.nd.array(batch_encoding_input)
            batch_decoding_input = mx.nd.array(batch_decoding_input)
            label_all = mx.nd.array(label)
            data_all = [batch_encoding_input, batch_decoding_input] + self.init_state_arrays
            data_names = ["data", "decoding_data"] + init_state_names
            label_names = ["softmax_label"]
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all, self.buckets[i_bucket])
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
        self.bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]
        




