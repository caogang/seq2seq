import mxnet as mx 
import numpy as np 
import data_utils 
from bucket_io import BucketSentenceIter
from lstm import seq2seq_lstm_unroll, seq2seq_lstm_softmax_stage, seq2seq_lstm_unroll_without_softmax
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

batch_size = 1
buckets = data_utils.BUCKETS
num_hidden = 1024
num_embed = 512
num_layers = 4
num_epoch = 300
learning_rate = 0.003
momentum = 0.0 
clip_norm = 1.0
seq_len = 30


def read_data(path, data, forward_data_feed):
    seq_size = 10
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

            if len(source_ids) < seq_size and len(target_ids) < seq_size - 1:
                data.append([source_ids, target_ids])

            first_line = second_line
            second_line = third_line
            third_line = read_file.readline()

            count += 1
            if count % 100000 == 0:
                logging.debug("Reading data line %d" % count)
    logging.info("Get sample %d pairs" % len(data))


def tokenize_id_with_seq_len(input_data, reverse=False):
        if type(input_data) == str:
            tokenized_id = data_utils.sentence_to_token_ids(input_data, vocab)
        else:
            tokenized_id = list(input_data)
        padding_data = [-1] * (seq_len - len(tokenized_id))
        if reverse:
            return np.array(list(reversed(tokenized_id + padding_data))).reshape(1,seq_len)
        else:
            return np.array(list(tokenized_id + padding_data)).reshape(1,seq_len)


if __name__ == "__main__":


    data_path = "../data/openSubtitle_truncate_100k.txt"
    target_path = "../data/tokenized_data_openSubtitle_100k.txt"
    vocabulary_path = "../data/vocabulary_openSubtitle_100k.txt"
    max_vocabulary_size = 30000
    data_utils.create_vocabulary(vocabulary_path, data_path, max_vocabulary_size)
    data_utils.data_to_token_ids(data_path, target_path, vocabulary_path)

    vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)
    num_vocab = len(rev_vocab)

    devs = mx.context.gpu(2)
    init_c = [("encode_init_c", (batch_size, num_layers, num_hidden))]
    init_h = [("encode_init_h", (batch_size, num_layers, num_hidden))]

    init_states = init_c + init_h
    encoding_data_shape = [("data", (batch_size,seq_len))]
    decoding_data_shape = [("decoding_data", (batch_size,seq_len))]
    softmax_label_shape = [("softmax_label", (batch_size, seq_len))]
    model_input_shape = dict(init_states + encoding_data_shape + decoding_data_shape)

    sym = seq2seq_lstm_unroll_without_softmax(seq_len, num_hidden, num_embed, num_vocab, num_layers)
    grad_req = dict([(key, "write") for key in sym.list_arguments()])
    for key in model_input_shape.keys():
        grad_req[key] = "null"
    grad_req["embed_weight"] = "null"
    model = sym.simple_bind(ctx= devs, grad_req = grad_req, **model_input_shape)
    model_arg_arrays = dict(zip(sym.list_arguments(), model.arg_arrays))
    init = mx.init.Uniform(0.07)
    for name, arr in model_arg_arrays.items():
        if name not in model_input_shape:
            init(name, arr)

    cost_sym = seq2seq_lstm_softmax_stage()
    cost_grad_req = {"softmax_label" : "null", "pred" : "write"}
    cost_input_shape = {"softmax_label":(batch_size, seq_len),
                        "pred":(batch_size * seq_len, num_vocab)}
    cost_model = cost_sym.simple_bind(ctx=devs, grad_req=cost_grad_req, **cost_input_shape)



    optimizer = mx.optimizer.SGD(momentum = momentum,
                                 learning_rate = learning_rate,
                                 clip_gradient = clip_norm)
    updater = mx.optimizer.get_updater(optimizer)
    
    data = []
    read_data(target_path, data, True)
    bucket_idx_all = np.random.permutation(len(data))
    for num in range(num_epoch):
        for i in range(len(data)):
            encoding_data, decoding_data = data[bucket_idx_all[i]]
            encoding_data_padded = tokenize_id_with_seq_len(encoding_data, reverse=True)
            label = tokenize_id_with_seq_len(decoding_data + [data_utils.EOS_ID], reverse=False)
            decoding_data_padded = np.zeros(label.shape)
            decoding_data_padded[:,:-1] = label[:,1:]
            decoding_data_padded[:,-1] = data_utils.GO_ID
            for name in model_input_shape.keys():
                model.arg_dict[name][:] = 0
            model.arg_dict["data"][:] = encoding_data_padded
            model.arg_dict["decoding_data"][:] = decoding_data_padded
            cost_model.arg_dict["softmax_label"][:] = label 
            model.forward(is_train=True)
            model.outputs[0].copyto(cost_model.arg_dict["pred"])
            # logging.info(model.outputs[0].asnumpy())
            cost_model.forward(is_train=True)
            cost_model.backward()
            model.backward(cost_model.grad_dict["pred"])
            for name, arr in model_arg_arrays.items():
                if name not in model_input_shape and name != "embed_weight":
                    # logging.info(name)
                    # logging.info(arr)
                    # logging.info(model.grad_dict[name].asnumpy())
                    updater(0, model.grad_dict[name], model.arg_dict[name])
            logging.info("Epoch[%d] sample [%d] complete" % (num+1,i + 1))
        
        weights = {}
        for key in sym.list_arguments():
            if key not in model_input_shape:
                weights[key] = self.model_arg_arrays[key]
        mx.model.save_checkpoint("../snapshots/test", num+1, sym , param_dict, model.aux_dict)








