import mxnet as mx 
import numpy as np 
import data_utils 
from seq2seq_model import Seq2SeqInferenceModel
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    batch_size = 1
    buckets = data_utils.BUCKETS
    num_hidden = 1024
    num_embed = 512

    momentum = 0.0 
    num_layer = 4
    learning_rate = 0.01
    beam_size = 10

    vocabulary_path = "../data/vocabulary_openSubtitle_100k.txt"
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)

    devs = mx.context.gpu(0)
    _, arg_params, __ = mx.model.load_checkpoint("../snapshots/forward_seq2seq", 20)
    model = Seq2SeqInferenceModel(30, batch_size, learning_rate, 
                vocab, rev_vocab, num_hidden, num_embed, num_layer, arg_params, beam_size, ctx=devs , dropout=0. )

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

        
