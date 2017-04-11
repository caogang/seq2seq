import numpy as np
import mxnet as mx
import data_utils
import sys
sys.path.append('../')
sys.path.append('./')
from lstm import seq2seq_lstm_unroll_without_softmax, seq2seq_lstm_softmax_stage
import logging

class PolicyGradientUpdateModel():
    def __init__(self, seq_len, batch_size, learning_rate, textData, num_hidden, num_embed,
                 num_layers, arg_params, ctx=mx.gpu(0), dropout=0.):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.textData = textData
        self.num_vocab = textData.getVocabularySize()
        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.devs = ctx
        self.num_layers = num_layers

        init_c = [('encode_init_c', (self.batch_size, self.num_layers, self.num_hidden))]
        init_h = [('encode_init_h', (self.batch_size, self.num_layers, self.num_hidden))]
        init_states = init_c + init_h
        encoding_data_shape = [("data", (batch_size, seq_len))]
        decoding_data_shape = [("decoding_data", (batch_size, seq_len))]
        self.model_input_shape = dict(init_states + encoding_data_shape + decoding_data_shape)
        self.model, self.model_arg_arrays = self._createModel(arg_params, **self.model_input_shape)
        cost_input_shape = dict(
            [("softmax_label", (batch_size, seq_len)), ("pred", (self.seq_len * self.batch_size, self.num_vocab))])
        self.cost_model, self.cost_arg_arrays = self._createCost(arg_params, **cost_input_shape)

        self.accumulate_weight = {}
        self.weights_key = []
        for key in self.model_symbol.list_arguments():
            if key not in self.model_input_shape:
                self.weights_key.append(key)

        optimizer = mx.optimizer.create('sgd',  # rescaled_grad = (1.0),
                                        learning_rate=self.learning_rate)
        self.updater = mx.optimizer.get_updater(optimizer)

    def forward(self, encoding_data, decoding_data):
        if type(encoding_data) == list and type(decoding_data) == list:
            encoding_data_padded = self.tokenize_id_with_seq_len(encoding_data, reverse=True)
            if decoding_data[-1] != data_utils.EOS_ID:
                decoding_data += [data_utils.EOS_ID]
            if len(decoding_data) + 1 >= self.seq_len:
                decoding_data_tmp = list([data_utils.GO_ID] + decoding_data)
                decoding_data_tmp = decoding_data_tmp[:self.seq_len]
                decoding_data_padded = self.tokenize_id_with_seq_len(decoding_data_tmp, reverse=False)
            else:
                decoding_data_padded = self.tokenize_id_with_seq_len(list([data_utils.GO_ID] + decoding_data))

        for key in self.model_input_shape.keys():
            self.model_arg_arrays[key][:] = 0
        encoding_data_padded.copyto(self.model_arg_arrays["data"])
        decoding_data_padded.copyto(self.model_arg_arrays["decoding_data"])
        self.model.forward(is_train=True)
        pred = self.model.outputs[0]
        label = np.zeros((self.batch_size, self.seq_len)) - 1
        label[:, :-1] = decoding_data_padded.asnumpy()[:, 1:]
        mx.nd.array(label).copyto(self.cost_arg_arrays["softmax_label"])
        pred.copyto(self.cost_arg_arrays["pred"])
        self.cost_model.forward(is_train=True)
        self.cost_model.backward()
        return self.cost_model.grad_dict["pred"].asnumpy()

    def initialize_dict(self, lhs, rhs, key):
        size = rhs[key].asnumpy().shape
        lhs[key] = np.zeros(size)
        lhs[key][:] += rhs[key].asnumpy()
        logging.info(key)
        logging.info(rhs[key].asnumpy())

    def backward(self, gradient):
        # logging.info(self.model.grad_dict["pred"].asnumpy().shape)
        # logging.info(self.model_symbol.list_arguments())
        self.model.backward(gradient)
        # logging.info(gradient.asnumpy())
        if len(self.accumulate_weight) == 0:
            for key in self.weights_key:
                self.initialize_dict(self.accumulate_weight, self.model.grad_dict, key)
        else:
            for key in self.weights_key:
                logging.info(key)
                logging.info(self.model.grad_dict[key].asnumpy())
                self.accumulate_weight[key][:] += self.model.grad_dict[key].asnumpy()

    def update_params(self):
        for i, key in enumerate(self.weights_key):
            logging.info(self.accumulate_weight[key])
            self.updater(i, mx.nd.array(self.accumulate_weight[key], ctx=self.devs), self.model.arg_dict[key])
        for key in self.weights_key:
            self.accumulate_weight[key][:] = 0.0

    def tokenize_id_with_seq_len(self, input_data, reverse=False):
        if type(input_data) == str:
            tokenized_id = self.textData.sentence2id(input_data)
        else:
            tokenized_id = list(input_data)
        padding_data = [self.textData.padToken] * (self.seq_len - len(tokenized_id))
        if reverse:
            return mx.nd.array(np.array(list(reversed(tokenized_id + padding_data))).reshape(1, self.seq_len))
        else:
            return mx.nd.array(np.array(list(tokenized_id + padding_data)).reshape(1, self.seq_len))

    def get_weights(self):
        """
        return the dict of weights that used in the model
        """
        weights = {}
        for key in self.model_symbol.list_arguments():
            if key not in self.model_input_shape:
                weights[key] = self.model_arg_arrays[key]
        return weights

    def _createModel(self, arg_params, **input_shapes):
        """
        create the main encoder-decoder model executor

        return:
        model: the executor of the encoder-decoder model
        arg_arrays: the dict of reference for model arguments
        """
        symbol = seq2seq_lstm_unroll_without_softmax(self.seq_len, self.num_hidden,
                                                     self.num_embed, self.num_vocab, self.num_layers)
        req = {key: "write" for key in symbol.list_arguments()}
        for key in input_shapes.keys(): req[key] = 'null'
        model = symbol.simple_bind(ctx=self.devs, grad_req=req, **input_shapes)
        arg_arrays = dict(zip(symbol.list_arguments(), model.arg_arrays))
        logging.info(symbol.list_arguments())
        logging.info(model.arg_arrays)

        self.model_symbol = symbol
        for name, arg in arg_params.items():
            if name in arg_arrays:
                arg_arrays[name][:] = arg
        return model, arg_arrays

    def _createCost(self, arg_params, **input_shapes):
        """
        create the softmaxOutput (cross Entropy) cost model
        """
        cost_sym = seq2seq_lstm_softmax_stage()
        req = {key: "write" for key in cost_sym.list_arguments()}
        req["softmax_label"] = 'null'
        cost_model = cost_sym.simple_bind(ctx=self.devs, grad_req=req, **input_shapes)
        arg_arrays = dict(zip(cost_sym.list_arguments(), cost_model.arg_arrays))

        self.cost_symbol = cost_sym
        for name, arg in arg_params.items():
            if name in arg_arrays:
                arg_arrays[name][:] = arg
        return cost_model, arg_arrays

    def save_weights(self, save_path, num_epoch):
        param_dict = self.get_weights()
        mx.model.save_checkpoint(save_path, num_epoch, self.model_symbol, param_dict, self.model.aux_dict)
