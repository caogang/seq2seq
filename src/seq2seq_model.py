import random
import numpy as np 
import mxnet as mx 
import data_utils 
from lstm import seq2seq_lstm_inference_encoding_symbol, seq2seq_lstm_inference_decoding_symbol
import logging

class sentence():
    def __init__(self, prev_sentence, cell, hidden, encoding_hidden, prob, prevProb, label):
        self.concatSentence = list(prev_sentence)
        self.concatSentence.append(label)
        self.cell = mx.nd.array(cell.asnumpy())
        self.hidden = mx.nd.array(hidden.asnumpy())
        self.encoding_hidden = mx.nd.array(encoding_hidden.asnumpy())
        self.currentLogProb = np.log(prob) 
        # logging.info(self.currentLogProb)
        #self.noise = np.random.uniform(-0.7,0.7)
        self.noise = 0
        self.noisedCumulativeLogProb = self.currentLogProb + prevProb + self.noise
        self.unnoised_log_Prob = self.currentLogProb + prevProb
        self.label = label 

    def get_cumulative_log_prob(self):
        return self.unnoised_log_Prob

    def get_noised_log_prob(self):
        return self.noisedCumulativeLogProb

    def get_cell(self):
        return self.cell 

    def get_hidden(self):
        return self.hidden

    def get_encoding_hidden(self):
        return self.encoding_hidden

    def get_label(self):
        return self.label 

    def get_concat_sentence(self):
        return self.concatSentence

    def __eq__(self, other):
        return self.get_noised_log_prob() == other.get_noised_log_prob()

    def __gt__(self, other):
        return self.get_noised_log_prob() > other.get_noised_log_prob()

    def __ne__(self, other):
        return not self == other 

    def __ge__(self, other):
        return self > other or self == other 

    def __lt__(self, other):
        return not self > other 

    def __le__(self, other):
        return self < other or self == other 

    def __cmp__(self, other):
        return cmp(self.get_noised_log_prob(), other.get_noised_log_prob())


class Seq2SeqInferenceModel():
    def __init__(self, seq_len, batch_size, learning_rate, 
                vocab, rev_vocab, num_hidden, num_embed, num_layer ,arg_params, beam_size = 0, ctx=mx.gpu(0), dropout=0.):
        self.seq_len = seq_len
        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.num_vocab = len(rev_vocab)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_hidden = num_hidden 
        self.num_embed = num_embed 
        self.num_layers = num_layer
        self.ctx = ctx
        self.dropout = dropout 
        self.beam_size = beam_size

        self.encoding_sym = seq2seq_lstm_inference_encoding_symbol(self.seq_len, self.num_hidden, 
                                                        self.num_embed,self.num_vocab, self.num_layers,dropout)
        init_c = [('encode_init_c', (self.batch_size, self.num_layers, self.num_hidden))]
        init_h = [('encode_init_h', (self.batch_size, self.num_layers, self.num_hidden))]
        init_states = init_c + init_h
        encoding_data_shape = [("data", (batch_size,seq_len))]
        encoding_input_shape = dict(init_states + encoding_data_shape)
        self.encoding_executor = self.encoding_sym.simple_bind(ctx= self.ctx, **encoding_input_shape)
        
        self.decoding_sym = seq2seq_lstm_inference_decoding_symbol(self.seq_len, self.num_hidden,
                                                        self.num_embed, self.num_vocab, self.num_layers, dropout)

        decoding_input_shape = {"decoding_data":(self.batch_size, 1),"decode_init_c":(self.num_layers,self.batch_size,num_hidden),"decode_init_h":(self.num_layers,self.batch_size,num_hidden)}
        self.decoding_executor = self.decoding_sym.simple_bind(ctx = self.ctx, **decoding_input_shape)
        self.load_params(arg_params)


        self.state_name = []
        self.state_name.append("encode_init_c")
        self.state_name.append("encode_init_h")

    def forward_encoding(self, input_data):
        input_data_tmp = self.tokenize_id_with_seq_len(input_data, reverse = True)
        input_data_tmp.copyto(self.encoding_executor.arg_dict["data"])
        for key in self.state_name:
            self.encoding_executor.arg_dict[key][:] = 0.
        self.encoding_executor.forward()
        next_hidden = self.encoding_executor.outputs[1]
        next_cell = self.encoding_executor.outputs[2]
        return next_hidden, next_cell

    def forward_greedy(self, input_data):
        next_hidden, next_cell = self.forward_encoding(input_data)
        self.decoding_executor.arg_dict["decoding_data"][:] = data_utils.GO_ID
        next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
        next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        # logging.info(self.encoding_executor.arg_dict["embed_weight"].asnumpy())
        # logging.info(self.encoding_executor.arg_dict["encode_weight"].asnumpy())
        result = []
        for seqidx in xrange(self.seq_len):
            self.decoding_executor.forward()
            next_hidden = self.decoding_executor.outputs[1]
            next_cell = self.decoding_executor.outputs[2]
            pred = np.argmax(self.decoding_executor.outputs[0].asnumpy())
            result.append(pred)
            self.decoding_executor.arg_dict["decoding_data"][:] = pred
            next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
            next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        return self.response(result)

    def forward_beam(self, input_data):
        encoding_hidden, encoding_cell = self.forward_encoding(input_data)
        self.decoding_executor.arg_dict["decoding_data"][:] = data_utils.GO_ID
        encoding_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
        encoding_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        # Beam search
        beam_size = self.beam_size
        beam = []
        result = []
        for seqidx in xrange(self.seq_len):
            if seqidx == 0:
                self.decoding_executor.forward()
                next_hidden = self.decoding_executor.outputs[1]
                next_cell = self.decoding_executor.outputs[2]
                softmax_output = self.decoding_executor.outputs[0].asnumpy()
                softmax_output = softmax_output.reshape((softmax_output.shape[1],))
                idx = np.argpartition(softmax_output, -beam_size)[-beam_size:]
                sftm = softmax_output[idx]
                for i in range(beam_size):
                    stc_tmp = sentence(list(), next_cell, next_hidden, encoding_hidden, sftm[i], 0, idx[i])
                    beam.append(stc_tmp)
            else:
                beam_tmp = []
                for i in range(beam_size):
                    stc_tmp = beam[i]
                    if data_utils.EOS_ID in stc_tmp.get_concat_sentence():
                        beam_tmp.append(stc_tmp)
                        continue
                    self.decoding_executor.arg_dict["decoding_data"][:] = stc_tmp.get_label()
                    stc_tmp.get_cell().copyto(self.decoding_executor.arg_dict["decode_init_c"])
                    stc_tmp.get_hidden().copyto(self.decoding_executor.arg_dict["decode_init_h"])
                    self.decoding_executor.forward()
                    next_hidden = self.decoding_executor.outputs[1]
                    next_cell = self.decoding_executor.outputs[2]
                    softmax_output = self.decoding_executor.outputs[0].asnumpy()
                    softmax_output = softmax_output.reshape((softmax_output.shape[1],))
                    idx = np.argpartition(softmax_output, -beam_size)[-beam_size:]
                    sftm = softmax_output[idx]
                    prev_sentence = stc_tmp.get_concat_sentence()
                    prev_prob = stc_tmp.get_cumulative_log_prob()
                    for i in range(beam_size):
                        stc = sentence(prev_sentence, next_cell, next_hidden, encoding_hidden, sftm[i], prev_prob, idx[i])
                        # logging.info(stc.get_concat_sentence())
                        # logging.info(stc.get_cumulative_log_prob())
                        beam_tmp.append(stc)
                beam = sorted(beam_tmp)[-beam_size:]
        return beam[::-1]

    def get_log_prob(self, encoding_data, decoding_data):
        """
        args:
        encoding_data: a mxnet ndarray of token ids of a sentence
        decoding_data: an numpy array of token ids of a sentence
        return:
        log_prob: log probability of the sentence
        """
        for key in self.state_name:
            self.encoding_executor.arg_dict[key][:] = 0.
        encoding_data.copyto(self.encoding_executor.arg_dict["data"])
        self.encoding_executor.forward()
        next_hidden = self.encoding_executor.outputs[1]

        next_cell = self.encoding_executor.outputs[2]
        self.decoding_executor.arg_dict["decoding_data"][:] = data_utils.GO_ID
        next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
        next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])
        result = []
        sftm = 0.0
        terminal = False
        for seqidx in xrange(self.seq_len):
            self.decoding_executor.forward()
            next_hidden = self.decoding_executor.outputs[1]
            next_cell = self.decoding_executor.outputs[2]
            sftm_tmp = self.decoding_executor.outputs[0].asnumpy()[0, decoding_data[0,seqidx]]
            sftm += np.log(sftm_tmp)
            if terminal:
                break
            if decoding_data[0, seqidx] == data_utils.EOS_ID:
                terminal = True
            self.decoding_executor.arg_dict["decoding_data"][:] = decoding_data[0, seqidx]
            next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
            next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        return sftm 

    def load_params(self, params):
        for key in self.encoding_executor.arg_dict.keys():
            if key in params:
                params[key].copyto(self.encoding_executor.arg_dict[key])
        for key in self.decoding_executor.arg_dict.keys():
            if key in params:
                params[key].copyto(self.decoding_executor.arg_dict[key])


    def response(self,result):
        response_tmp = []
        for i in result:
            if i == data_utils.PAD_ID:
                pass
            elif i == data_utils.EOS_ID:
                response_tmp.append(self.rev_vocab[i])
                break
            elif i < len(self.rev_vocab):
                response_tmp.append(self.rev_vocab[i])
            else:
                response_tmp.append(data_utils._UNK)
        response_str = " ".join(response_tmp)
        return response_str

    def tokenize_id_with_seq_len(self,input_data, reverse=False):
        if type(input_data) == str:
            tokenized_id = data_utils.sentence_to_token_ids(input_data, self.vocab)
        else:
            tokenized_id = list(input_data)
        padding_data = [-1] * (self.seq_len - len(tokenized_id))
        if reverse:
            return mx.nd.array(np.array(list(reversed(tokenized_id + padding_data))).reshape(1,self.seq_len))
        else:
            return mx.nd.array(np.array(list(tokenized_id + padding_data)).reshape(1,self.seq_len))


class Seq2SeqInferenceModelCornellData():
    def __init__(self, seq_len, batch_size, learning_rate,
                 textData, num_hidden, num_embed, num_layer, arg_params, beam_size=0, ctx=mx.gpu(0),
                 dropout=0.):
        self.seq_len = seq_len
        self.textData = textData
        self.num_vocab = textData.getVocabularySize()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.num_layers = num_layer
        self.ctx = ctx
        self.dropout = dropout
        self.beam_size = beam_size

        self.encoding_sym = seq2seq_lstm_inference_encoding_symbol(self.seq_len, self.num_hidden,
                                                                   self.num_embed, self.num_vocab, self.num_layers,
                                                                   dropout)
        init_c = [('encode_init_c', (self.batch_size, self.num_layers, self.num_hidden))]
        init_h = [('encode_init_h', (self.batch_size, self.num_layers, self.num_hidden))]
        init_states = init_c + init_h
        encoding_data_shape = [("data", (batch_size, seq_len))]
        encoding_input_shape = dict(init_states + encoding_data_shape)
        self.encoding_executor = self.encoding_sym.simple_bind(ctx=self.ctx, **encoding_input_shape)

        self.decoding_sym = seq2seq_lstm_inference_decoding_symbol(self.seq_len + 2, self.num_hidden,
                                                                   self.num_embed, self.num_vocab, self.num_layers,
                                                                   dropout)

        decoding_input_shape = {"decoding_data": (self.batch_size, 1),
                                "decode_init_c": (self.num_layers, self.batch_size, num_hidden),
                                "decode_init_h": (self.num_layers, self.batch_size, num_hidden)}
        self.decoding_executor = self.decoding_sym.simple_bind(ctx=self.ctx, **decoding_input_shape)
        self.load_params(arg_params)

        self.state_name = []
        self.state_name.append("encode_init_c")
        self.state_name.append("encode_init_h")

    def forward_encoding(self, input_data):
        input_data_tmp = self.tokenize_id_with_seq_len(input_data, reverse=True)
        input_data_tmp.copyto(self.encoding_executor.arg_dict["data"])
        for key in self.state_name:
            self.encoding_executor.arg_dict[key][:] = 0.
        self.encoding_executor.forward()
        next_hidden = self.encoding_executor.outputs[1]
        next_cell = self.encoding_executor.outputs[2]
        return next_hidden, next_cell

    def forward_greedy(self, input_data):
        next_hidden, next_cell = self.forward_encoding(input_data)
        self.decoding_executor.arg_dict["decoding_data"][:] = self.textData.goToken
        next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
        next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        # logging.info(self.encoding_executor.arg_dict["embed_weight"].asnumpy())
        # logging.info(self.encoding_executor.arg_dict["encode_weight"].asnumpy())
        result = []
        for seqidx in xrange(self.seq_len + 2):
            self.decoding_executor.forward()
            next_hidden = self.decoding_executor.outputs[1]
            next_cell = self.decoding_executor.outputs[2]
            pred = np.argmax(self.decoding_executor.outputs[0].asnumpy())
            result.append(pred)
            self.decoding_executor.arg_dict["decoding_data"][:] = pred
            next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
            next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        return self.response(result)

    def __random_pick(self, some_list, probabilities):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        probabilities = probabilities.reshape((-1,)).tolist()
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item

    def forward_sample(self, input_data):
        next_hidden, next_cell = self.forward_encoding(input_data)
        self.decoding_executor.arg_dict["decoding_data"][:] = self.textData.goToken
        next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
        next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        # logging.info(self.encoding_executor.arg_dict["embed_weight"].asnumpy())
        # logging.info(self.encoding_executor.arg_dict["encode_weight"].asnumpy())
        result = []
        for seqidx in xrange(self.seq_len + 2):
            self.decoding_executor.forward()
            next_hidden = self.decoding_executor.outputs[1]
            next_cell = self.decoding_executor.outputs[2]

            pred = self.__random_pick(range(0, self.textData.getVocabularySize()),
                                      self.decoding_executor.outputs[0].asnumpy())

            result.append(pred)
            self.decoding_executor.arg_dict["decoding_data"][:] = pred
            next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
            next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        return self.response(result)

    def forward_beam(self, input_data):
        encoding_hidden, encoding_cell = self.forward_encoding(input_data)
        self.decoding_executor.arg_dict["decoding_data"][:] = self.textData.goToken
        encoding_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
        encoding_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        # Beam search
        beam_size = self.beam_size
        beam = []
        result = []
        for seqidx in xrange(self.seq_len + 2):
            if seqidx == 0:
                self.decoding_executor.forward()
                next_hidden = self.decoding_executor.outputs[1]
                next_cell = self.decoding_executor.outputs[2]
                softmax_output = self.decoding_executor.outputs[0].asnumpy()
                softmax_output = softmax_output.reshape((softmax_output.shape[1],))
                idx = np.argpartition(softmax_output, -beam_size)[-beam_size:]
                sftm = softmax_output[idx]
                for i in range(beam_size):
                    stc_tmp = sentence(list(), next_cell, next_hidden, encoding_hidden, sftm[i], 0, idx[i])
                    beam.append(stc_tmp)
            else:
                beam_tmp = []
                for i in range(beam_size):
                    stc_tmp = beam[i]
                    if self.textData.eosToken in stc_tmp.get_concat_sentence():
                        beam_tmp.append(stc_tmp)
                        continue
                    self.decoding_executor.arg_dict["decoding_data"][:] = stc_tmp.get_label()
                    stc_tmp.get_cell().copyto(self.decoding_executor.arg_dict["decode_init_c"])
                    stc_tmp.get_hidden().copyto(self.decoding_executor.arg_dict["decode_init_h"])
                    self.decoding_executor.forward()
                    next_hidden = self.decoding_executor.outputs[1]
                    next_cell = self.decoding_executor.outputs[2]
                    softmax_output = self.decoding_executor.outputs[0].asnumpy()
                    softmax_output = softmax_output.reshape((softmax_output.shape[1],))
                    idx = np.argpartition(softmax_output, -beam_size)[-beam_size:]
                    sftm = softmax_output[idx]
                    prev_sentence = stc_tmp.get_concat_sentence()
                    prev_prob = stc_tmp.get_cumulative_log_prob()
                    for i in range(beam_size):
                        stc = sentence(prev_sentence, next_cell, next_hidden, encoding_hidden, sftm[i], prev_prob,
                                       idx[i])
                        # logging.info(stc.get_concat_sentence())
                        # logging.info(stc.get_cumulative_log_prob())
                        beam_tmp.append(stc)
                beam = sorted(beam_tmp)[-beam_size:]
        return beam[::-1]

    def get_log_prob(self, encoding_data, decoding_data):
        """
        args:
        encoding_data: a mxnet ndarray of token ids of a sentence
        decoding_data: an numpy array of token ids of a sentence
        return:
        log_prob: log probability of the sentence
        """
        for key in self.state_name:
            self.encoding_executor.arg_dict[key][:] = 0.
        encoding_data.copyto(self.encoding_executor.arg_dict["data"])
        self.encoding_executor.forward()
        next_hidden = self.encoding_executor.outputs[1]

        next_cell = self.encoding_executor.outputs[2]
        self.decoding_executor.arg_dict["decoding_data"][:] = self.textData.goToken
        next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
        next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])
        result = []
        sftm = 0.0
        terminal = False
        for seqidx in xrange(self.seq_len + 2):
            self.decoding_executor.forward()
            next_hidden = self.decoding_executor.outputs[1]
            next_cell = self.decoding_executor.outputs[2]
            sftm_tmp = self.decoding_executor.outputs[0].asnumpy()[0, decoding_data[0, seqidx]]
            sftm += np.log(sftm_tmp)
            if terminal:
                break
            if decoding_data[0, seqidx] == self.textData.eosToken:
                terminal = True
            self.decoding_executor.arg_dict["decoding_data"][:] = decoding_data[0, seqidx]
            next_cell.copyto(self.decoding_executor.arg_dict["decode_init_c"])
            next_hidden.copyto(self.decoding_executor.arg_dict["decode_init_h"])

        return sftm

    def load_params(self, params):
        for key in self.encoding_executor.arg_dict.keys():
            if key in params:
                params[key].copyto(self.encoding_executor.arg_dict[key])
        for key in self.decoding_executor.arg_dict.keys():
            if key in params:
                params[key].copyto(self.decoding_executor.arg_dict[key])

    def response(self, result):
        return self.textData.sequence2str(result)

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
