import mxnet as mx
import re
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

    devs = mx.context.gpu(2)

    iterations = args.epochRL
    d_steps = args.dStepsRL
    g_steps = args.gStepsRL
    save_epoch = args.saveEveryRL
    adv_prefix = args.loadPrefixAdv
    adv_load_epoch = args.loadEpochAdv
    _, g_arg_params, __ = mx.model.load_checkpoint(adv_prefix + '_g', adv_load_epoch)
    print 'loaded parameters....'
    inference_model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                                                       textData, num_hidden, num_embed, num_layer, g_arg_params,
                                                       beam_size,
                                                       ctx=devs, dropout=0.)
    print 'seq2seq loaded'
    #prefix = "../snapshots/discriminator-new-optimizer"
    #prefix = "../snapshots/discriminator"
    discriminator_model = HierarchyDiscriminatorModel(args, textData, ctx=devs, is_train=False,
                                                      prefix=adv_prefix+'_d', load_epoch=adv_load_epoch)
    print 'discriminator loaded'

    policy_gradient_model = PolicyGradientUpdateModel(args.maxLength, batch_size, learning_rate,
                                                      textData, num_hidden, num_embed, num_layer, g_arg_params)
    print 'policy gradient loaded'

    pattern = re.compile(r'<.*>')

    for i in xrange(1, iterations + 1):
        logging.info('Discriminator Training')
        for d in xrange(d_steps):
            sample_qa = textData.get_random_qapair()
            q = sample_qa[0]
            a = sample_qa[1]
            #a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
            a_machine = inference_model.forward_sample(q)
            a_machine = a_machine.rstrip(' <pad>')
            a_machine = a_machine.rstrip(' <eos>')
            positive_batch = (q, a, 1)
            negative_batch = (q, a_machine, 0)
            a_machine_list = a_machine.split(' ')
            extra_num = len(a_machine_list) - [pattern.match(x) for x in a_machine_list].count(None)

            while extra_num > 0 or len(a_machine_list) > args.maxLength + 0 or len(a_machine_list) <= 0:
                if len(a_machine_list) > args.maxLength + 0:
                    logging.info('Out of max length')
                    break
                if len(a_machine_list) <= 0:
                    logging.info('Too short')
                if extra_num > 0:
                    logging.info('Containing <*> string')
                sample_qa = textData.get_random_qapair()
                q = sample_qa[0]
                a = sample_qa[1]
                #a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
                a_machine = inference_model.forward_sample(q)
                a_machine = a_machine.rstrip(' <pad>')
                a_machine = a_machine.rstrip(' <eos>')
                positive_batch = (q, a, 1)
                negative_batch = (q, a_machine, 0)
                a_machine_list = a_machine.split(' ')
                extra_num = len(a_machine_list) - [pattern.match(x) for x in a_machine_list].count(None)

            discriminator_model.train_one_batch(positive_batch)
            discriminator_model.train_one_batch(negative_batch)
        logging.info('Generator Training')
        for g in xrange(g_steps):
            sample_qa = textData.get_random_qapair()
            q = sample_qa[0]
            a = sample_qa[1]
            # a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
            a_machine = inference_model.forward_sample(q)
            a_machine = a_machine.rstrip(' <pad>')
            a_machine = a_machine.rstrip(' <eos>')
            a_machine_list = a_machine.split(' ')
            extra_num = len(a_machine_list) - [pattern.match(x) for x in a_machine_list].count(None)

            while extra_num > 0 or len(a_machine_list) > args.maxLength + 0 or len(a_machine_list) <= 0:
                if len(a_machine_list) > args.maxLength + 0:
                    logging.info('Out of max length')
                    #pred_grad = policy_gradient_model.forward(q, a_machine)
                    #gradient = mx.nd.array(pred_grad * 0.1, ctx=devs)
                    #policy_gradient_model.backward(gradient)
                    #policy_gradient_model.update_params()
                    break
                if len(a_machine_list) <= 0:
                    logging.info('Too short')
                if extra_num > 0:
                    logging.info('Containing <*> string')
                sample_qa = textData.get_random_qapair()
                q = sample_qa[0]
                a = sample_qa[1]
                # a_machine = inference_model.response(inference_model.forward_beam(q)[0].get_concat_sentence())
                a_machine = inference_model.forward_sample(q)
                a_machine = a_machine.rstrip(' <pad>')
                a_machine = a_machine.rstrip(' <eos>')
                a_machine_list = a_machine.split(' ')
                extra_num = len(a_machine_list) - [pattern.match(x) for x in a_machine_list].count(None)

            logging.info('iteration : ' + str(i))
            logging.info('Q : ' + q + ' H : ' + a)
            logging.info('Q : ' + q + ' M : ' + a_machine)
            reward = discriminator_model.predict(q, a_machine)[1]
            logging.info('Probability Human : ' + str(reward))
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
            discriminator_model.save_check_points(adv_prefix + '_d', i)
            policy_gradient_model.save_weights(adv_prefix + '_g', i)
