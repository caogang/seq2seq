import sys
sys.path.insert(0, "../../python")
import mxnet as mx 
import numpy as np 
from collections import namedtuple 
import time 
import math 
import logging


def seq2seq_lstm_unroll(seq_len, num_hidden, num_embed, num_vocab, num_layer, dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    last_states = []
    encode_init_c = mx.sym.Variable("encode_init_c")
    encode_init_c = mx.sym.Reshape(data=encode_init_c, shape=(num_layer, -1, num_hidden))
    encode_init_h = mx.sym.Variable("encode_init_h")
    encode_init_h = mx.sym.Reshape(data=encode_init_h, shape=(num_layer, -1, num_hidden))
    encode_param = mx.sym.Variable("encode_weight")
    decode_param = mx.sym.Variable("decode_weight")

    # embedding layer
    encoding_data = mx.sym.Variable('data')
    encoding_data = mx.sym.transpose(encoding_data)
    decoding_data = mx.sym.Variable('decoding_data')
    decoding_data = mx.sym.transpose(decoding_data)
    label = mx.sym.Variable('softmax_label')

    encoding_embed = mx.sym.Embedding(data=encoding_data, input_dim=num_vocab,
                             weight=embed_weight, output_dim=num_embed, name='encoding_embed')
    decoding_embed = mx.sym.Embedding(data=decoding_data,input_dim=num_vocab,
                             weight=embed_weight, output_dim=num_embed, name='decoding_embed')

    output = mx.sym.RNN(data = encoding_embed, parameters = encode_param,
                              state = encode_init_h, state_cell = encode_init_c,
                              state_size=num_hidden, num_layers = num_layer, 
                              state_outputs=True, bidirectional=False, 
                              mode='lstm', name="encodeLSTM")

    o = output[0]
    #o = mx.sym.Stats(data=o, source_name='outputs_stats')
    h = mx.sym.Select(*[o, output[1]], index=1)
    c = output[2]
    #h = mx.sym.Stats(data=h, source_name='hidden_stats')
    #c = mx.sym.Stats(data=c, source_name='cell_stats')

    result = mx.sym.RNN(data = decoding_embed, parameters = decode_param,
                        state = h, state_cell = c,
                        state_size = num_hidden, num_layers = num_layer,
                        state_outputs=False, bidirectional=False,
                        mode='lstm', name='decodeLSTM')
    
    
    result = mx.sym.Reshape(data=result, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=result, num_hidden=num_vocab,
                                 weight=cls_weight, bias=cls_bias, name='pred')



    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, shape=(-1,))

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax', use_ignore = True)
    sm_reshape = mx.sym.Reshape(data=sm, shape=(seq_len + 2, -1, num_vocab))
    return mx.sym.SwapAxis(data=sm_reshape, dim1=0, dim2=1)


def seq2seq_lstm_unroll_without_softmax(seq_len, num_hidden, num_embed, num_vocab, num_layer, dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    last_states = []
    encode_init_c = mx.sym.Variable("encode_init_c")
    encode_init_c = mx.sym.Reshape(data=encode_init_c, shape=(num_layer, -1, num_hidden))
    encode_init_h = mx.sym.Variable("encode_init_h")
    encode_init_h = mx.sym.Reshape(data=encode_init_h, shape=(num_layer, -1, num_hidden))
    encode_param = mx.sym.Variable("encode_weight")
    decode_param = mx.sym.Variable("decode_weight")

    # embedding layer
    encoding_data = mx.sym.Variable('data')
    encoding_data = mx.sym.transpose(encoding_data)
    decoding_data = mx.sym.Variable('decoding_data')
    decoding_data = mx.sym.transpose(decoding_data)
    #label = mx.sym.Variable('softmax_label')

    encoding_embed = mx.sym.Embedding(data=encoding_data, input_dim=num_vocab,
                             weight=embed_weight, output_dim=num_embed, name='encoding_embed')
    # encoding_embed = mx.sym.Stats(data=encoding_embed, source_name='encoding_embed_state')
    decoding_embed = mx.sym.Embedding(data=decoding_data,input_dim=num_vocab,
                             weight=embed_weight, output_dim=num_embed, name='decoding_embed')

    output = mx.sym.RNN(data = encoding_embed, parameters = encode_param,
                              state = encode_init_h, state_cell = encode_init_c,
                              state_size=num_hidden, num_layers = num_layer, 
                              state_outputs=True, bidirectional=False, 
                              mode='lstm', name="encodeLSTM")

    o = output[0]
    # o = mx.sym.Stats(data=o, source_name='outputs_stats')
    h = mx.sym.Select(*[o, output[1]], index=1)
    c = output[2]
    # h = mx.sym.Stats(data=h, source_name='hidden_stats')
    # c = mx.sym.Stats(data=c, source_name='cell_stats')

    output = mx.sym.RNN(data = decoding_embed, parameters = decode_param,
                        state = h, state_cell = c,
                        state_size = num_hidden, num_layers = num_layer,
                        state_outputs=False, bidirectional=False,
                        mode='lstm', name='decodeLSTM')
    
    # output = mx.sym.Stats(data=output, source_name="decoding_output_stats")
    result = mx.sym.Reshape(data=output, shape=(-1, num_hidden))
    # result = mx.sym.Stats(data=result, source_name="result_reshape_stats")
    pred = mx.sym.FullyConnected(data=result, num_hidden=num_vocab,
                                 weight=cls_weight, bias=cls_bias, name='pred')



    # label = mx.sym.transpose(data=label)
    # label = mx.sym.Reshape(data=label, shape=(-1,))

    # sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax', use_ignore = True)
    # sm_reshape = mx.sym.Reshape(data=sm, shape=(seq_len, -1, num_vocab))
    # return mx.sym.SwapAxis(data=sm_reshape, dim1=0, dim2=1)
    return pred

def seq2seq_lstm_softmax_stage():
    label = mx.sym.Variable('softmax_label')
    pred = mx.sym.Variable('pred')
    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name="softmax", use_ignore = True)
    # sm_reshape = mx.sym.Reshape(data=sm, shape=(seq_len, -1, num_vocab))
    # return mx.sym.SwapAxis(data=sm_reshape, dim1=0, dim2=1)
    return sm




def seq2seq_lstm_inference_encoding_symbol(seq_len, num_hidden, num_embed, num_vocab, num_layer,dropout=0.):
    embed_weight = mx.sym.Variable("embed_weight")
    encode_init_c = mx.sym.Variable("encode_init_c")
    encode_init_c = mx.sym.Reshape(data=encode_init_c, shape=(num_layer, -1, num_hidden))
    encode_init_h = mx.sym.Variable("encode_init_h")
    encode_init_h = mx.sym.Reshape(data=encode_init_h, shape=(num_layer, -1, num_hidden))
    encode_param = mx.sym.Variable("encode_weight")
    encoding_data = mx.sym.Variable('data')
    encoding_data = mx.sym.transpose(encoding_data)

    encoding_embed = mx.sym.Embedding(data=encoding_data, input_dim=num_vocab,
                             weight=embed_weight, output_dim=num_embed, name='encoding_embed')

    output = mx.sym.RNN(data = encoding_embed, parameters = encode_param,
                              state = encode_init_h, state_cell = encode_init_c,
                              state_size=num_hidden, num_layers = num_layer, 
                              state_outputs=True, bidirectional=False, 
                              mode='lstm', name="encodeLSTM")

    o = output[0]
    h = output[1]
    c = output[2]
    return mx.sym.Group([o, h, c])

def seq2seq_lstm_inference_decoding_symbol(seq_len, num_hidden, num_embed, num_vocab, num_layer,dropout = 0.):
    """
    decoding_input: one word input
    """
    seqidx = 0
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    
    decode_init_h = mx.sym.Variable("decode_init_h")
    decode_init_h = mx.sym.Reshape(data=decode_init_h, shape=(num_layer, -1, num_hidden))
    decode_init_c = mx.sym.Variable("decode_init_c")
    decode_init_c = mx.sym.Reshape(data=decode_init_c, shape=(num_layer, -1, num_hidden))

    decoding_data = mx.sym.Variable('decoding_data')
    decoding_embed = mx.sym.Embedding(data=decoding_data,input_dim=num_vocab,
                             weight=embed_weight, output_dim=num_embed, name='decoding_embed')
    decode_param = mx.sym.Variable("decode_weight")

    output = mx.sym.RNN(data = decoding_embed, parameters = decode_param,
                        state = decode_init_h, state_cell = decode_init_c,
                        state_size = num_hidden, num_layers = num_layer,
                        state_outputs=True, bidirectional=False,
                        mode='lstm', name='decodeLSTM')

    result = mx.sym.Reshape(data=output[0], shape=(-1, num_hidden))
    h = output[1]
    c = output[2]
    pred = mx.sym.FullyConnected(data=result, num_hidden=num_vocab,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    pred = mx.sym.Softmax(data=pred, name="softmax_pred")

    return mx.sym.Group([pred, h, c])
