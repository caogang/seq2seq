import mxnet as mx
import numpy as np
import data_utils
from seq2seq_model import Seq2SeqInferenceModelCornellData
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from params import getArgs
from textdata import TextData

if __name__ == "__main__":
    args = getArgs()

    batch_size = 1
    buckets = data_utils.BUCKETS
    num_hidden = args.hiddenSize
    num_embed = args.embeddingSize

    momentum = 0.0
    num_layer = args.numLayers
    learning_rate = args.learningRate
    beam_size = 5#10

    textData = TextData(args)
    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2

    if args.load is None:
        args.load = 50

    devs = mx.context.gpu(1)
    #_, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
    _, arg_params, __ = mx.model.load_checkpoint("../snapshots/policy_gradient_g", 23500)
    model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                textData, num_hidden, num_embed, num_layer, arg_params, beam_size, ctx=devs, dropout=0. )

    try:
        fsock = open("../data/cornell_test.txt", "r")
    except IOError:
        print "The file don't exist, Please double check!"
        exit()
    inputs = fsock.readlines()
    fsock.close()
    output = ""
    for conversation_input in inputs:
        output += ">> "
        output += conversation_input
        output += "greedy:\n"
        output += model.forward_greedy(conversation_input)
        output += "\n"
        output += "beam search:\n"
        #for response in model.forward_beam(conversation_input):
            # logger.info(response.get_concat_sentence())
            # logger.info(model.response(response.get_concat_sentence()))
        output += model.response(model.forward_beam(conversation_input)[0].get_concat_sentence())
        output += "\n\n"
    print output
