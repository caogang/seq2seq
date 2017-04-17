import mxnet as mx
from seq2seq_model import Seq2SeqInferenceModelCornellData
from params import getArgs
from textdata import TextData

from discriminator.HierarchyDiscriminator import HierarchyDiscriminatorModel
from rl.policy_gradient_model import PolicyGradientUpdateModel

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = getArgs()

    batch_size = 1
    num_hidden = args.hiddenSize
    num_embed = args.embeddingSize

    momentum = 0.0
    num_layer = args.numLayers
    learning_rate = args.learningRate
    beam_size = 5  # 10

    textData = TextData(args)
    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2

    devs = mx.context.gpu(0)

    iterations = 100
    d_steps = 5
    g_steps = 1

    _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
    inference_model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                                                       textData, num_hidden, num_embed, num_layer, arg_params,
                                                       beam_size,
                                                       ctx=devs, dropout=0.)

    discriminator_model = HierarchyDiscriminatorModel(args, textData, is_train=False)

    policy_gradient_model = PolicyGradientUpdateModel(args.maxLength, batch_size, learning_rate,
                                                      textData, num_hidden, num_embed, num_layer, arg_params)

    for i in xrange(iterations):
        for d in xrange(d_steps):
            sample_qa = textData.get_random_qapair()
            q = sample_qa[0]
            a = sample_qa[1]
            a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
            positive_batch = (q, a, 1)
            negetive_batch = (q, a_machine, 0)
            discriminator_model.train_one_batch(positive_batch)
            discriminator_model.train_one_batch(negetive_batch)
        for g in xrange(g_steps):
            sample_qa = textData.get_random_qapair()
            q = sample_qa[0]
            a = sample_qa[1]
            a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
            reward = discriminator_model.predict(q, a_machine)[1]
            print reward


