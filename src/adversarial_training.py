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

    iterations = args.epochRL
    d_steps = args.dStepsRL
    g_steps = args.gStepsRL
    save_epoch = args.saveEveryRL

    _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
    inference_model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                                                       textData, num_hidden, num_embed, num_layer, arg_params,
                                                       beam_size,
                                                       ctx=devs, dropout=0.)

    prefix = "../snapshots/discriminator-new-optimizer"
    discriminator_model = HierarchyDiscriminatorModel(args, textData, ctx=devs, is_train=False, prefix=prefix)

    policy_gradient_model = PolicyGradientUpdateModel(args.maxLength, batch_size, learning_rate,
                                                      textData, num_hidden, num_embed, num_layer, arg_params)

    for i in xrange(1, iterations + 1):
        for d in xrange(d_steps):
            sample_qa = textData.get_random_qapair()
            q = sample_qa[0]
            a = sample_qa[1]
            a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
            positive_batch = (q, a, 1)
            negative_batch = (q, a_machine, 0)
            discriminator_model.train_one_batch(positive_batch)
            discriminator_model.train_one_batch(negative_batch)
        for g in xrange(g_steps):
            sample_qa = textData.get_random_qapair()
            q = sample_qa[0]
            a = sample_qa[1]
            a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
            print 'iteration : ' + str(i)
            print 'Q : ' + q + ' H : ' + a
            print 'Q : ' + q + ' M : ' + a_machine
            reward = discriminator_model.predict(q, a_machine)[1]
            print 'Probability Human : ' + str(reward)
            pred_grad = policy_gradient_model.forward(q, a_machine)
            gradient = mx.nd.array(pred_grad * reward, ctx=devs)
            policy_gradient_model.backward(gradient)
            policy_gradient_model.update_params()

            # teacher forcing
            pred_grad = policy_gradient_model.forward(q, a)
            gradient = mx.nd.array(pred_grad, ctx=devs)
            policy_gradient_model.backward(gradient)
            policy_gradient_model.update_params()
            
            inference_model.load_params(policy_gradient_model.get_weights())

        if i % save_epoch == 0:
            discriminator_model.save_check_points('../snapshots/policy_gradient_d', i)
            policy_gradient_model.save_weights('../snapshots/policy_gradient_g', i)
