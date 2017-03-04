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
    beam_size = 10

    textData = TextData(args)
    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2

    devs = mx.context.gpu(0)
    _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", 30)
    model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                textData, num_hidden, num_embed, num_layer, arg_params, beam_size, ctx=devs, dropout=0. )

    while True:
        conversation_input = raw_input("You >> ")
        if conversation_input.lower() == "quit":
            break
        logger.info("greedy:")
        logger.info(model.forward_greedy(conversation_input))
        logger.info("beam search:")
        for response in model.forward_beam(conversation_input):
            # logger.info(response.get_concat_sentence())
            logger.info(model.response(response.get_concat_sentence()))
