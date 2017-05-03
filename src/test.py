import mxnet as mx
import numpy as np
import data_utils
from bucket_io import BucketSentenceIter
from lstm import seq2seq_lstm_unroll
from seq2seq_model import Seq2SeqInferenceModelCornellData

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from textdata import TextData, CornellDataIter
from params import getArgs


class GroupAccuracy(mx.metric.EvalMetric):
    """Calculate group accuracy."""

    def __init__(self):
        super(GroupAccuracy, self).__init__('group accuracy')

    def update(self, labels, preds):
        preds = [preds[0]]
        mx.metric.check_label_shapes(labels, preds)
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = self.ndarray_argmax(pred_label)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            #print label, pred_label
            mx.metric.check_label_shapes(label, pred_label)
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

    def ndarray_argmax(self, pred):  # argmax the last dim
        if len(pred.shape) > 2:
            flatten_shape = (np.prod(pred.shape[0:-1]), pred.shape[-1])
            pred_label = mx.ndarray.argmax_channel(pred.reshape(flatten_shape))
            return pred_label.reshape(pred.shape[0:-1])
        else:
            return mx.ndarray.argmax_channel(pred)

class GroupPerplexity(mx.metric.EvalMetric):
    """Calculate group perplexity."""
    def __init__(self, ignore_label, axis=-1):
        super(GroupPerplexity, self).__init__('Group Perplexity')
        self.ignore_label = ignore_label
        self.axis = axis

    def update(self, labels, preds):
        preds = [preds[0]]
        assert len(labels) == len(preds)
        loss = 0.
        num = 0
        probs = []

        for label, pred in zip(labels, preds):
            assert label.size == pred.size/pred.shape[-1], \
                "shape mismatch: %s vs. %s"%(label.shape, pred.shape)
            label = label.as_in_context(pred.context).astype(dtype='int32').reshape((label.size,))
            pred = mx.ndarray.pick(pred.reshape((-1, pred.shape[-1])), label, axis=self.axis)
            probs.append(pred)

        for label, prob in zip(labels, probs):
            prob = prob.asnumpy()
            if self.ignore_label is not None:
                ignore = label.asnumpy().flatten() == self.ignore_label
                prob = prob*(1-ignore) + ignore
                num += prob.size - ignore.sum()
            else:
                num += prob.size
            loss += -np.log(np.maximum(1e-10, prob)).sum()

        self.sum_metric += np.exp(loss / num)
        self.num_inst += 1

if __name__ == "__main__":
    #args = parser.parse_args()
    args = getArgs()
    batch_size = args.batchSize
    buckets = [args.maxLength]
    num_hidden = args.hiddenSize
    num_embed = args.embeddingSize
    num_layers = args.numLayers
    num_epoch = args.numEpochs
    learning_rate = args.learningRate
    momentum = 0.0
    clip_norm = 1.0

    textData = TextData(args)
    args.maxLengthEnco = args.maxLength
    args.maxLengthDeco = args.maxLength + 2

    textData.shuffle()
    devs = mx.context.gpu(2)

    # Put needed training stage here
    test_stage = [1]

    if 1 in test_stage:
        def sym_gen(seq_len):
            return seq2seq_lstm_unroll(seq_len, num_hidden, num_embed,
                    num_vocab=textData.getVocabularySize(), num_layer = num_layers, dropout=args.dropout)
        forward_seq2seq_sym = sym_gen

        init_c = [("encode_init_c", (batch_size, num_layers, num_hidden))]
        init_h = [("encode_init_h", (batch_size, num_layers, num_hidden))]
        init_states = init_c + init_h

        Forward_data_feed = True
        forward_data_eval = CornellDataIter(textData, [args.maxLength,], args.batchSize, init_states, True, validation=True)
                                        #BucketSentenceIter(target_path, vocab,
                                        #buckets, batch_size,num_layers, init_states, Forward_data_feed)

        optimizer = mx.optimizer.SGD(momentum = momentum,
                                     learning_rate = learning_rate,
                                     clip_gradient = clip_norm)

        model = mx.model.FeedForward.load("../snapshots/policy_gradient_g", 1, ctx=devs)
        #model = mx.model.FeedForward.load("../snapshots/seq2seq_newdata", 26, ctx=devs)
        model.score(forward_data_eval, GroupPerplexity(None), display=True)
        model = mx.model.FeedForward.load("../snapshots/policy_gradient_g", 100, ctx=devs)
        model.score(forward_data_eval, GroupPerplexity(None), display=True)
        model = mx.model.FeedForward.load("../snapshots/policy_gradient_g", 12000, ctx=devs)
        model.score(forward_data_eval, GroupPerplexity(None), display=True)
        model = mx.model.FeedForward.load("../snapshots/policy_gradient_g", 14000, ctx=devs)
        model.score(forward_data_eval, GroupPerplexity(None), display=True)
    elif 2 in test_stage:

        iterations = 100

        num_layer = args.numLayers
        learning_rate = args.learningRate
        beam_size = 5  # 10

        _, arg_params, __ = mx.model.load_checkpoint("../snapshots/policy_gradient_g", 12000)
        model = Seq2SeqInferenceModelCornellData(args.maxLength, batch_size, learning_rate,
                                                 textData, num_hidden, num_embed, num_layer, arg_params, beam_size,
                                                 ctx=devs, dropout=0.)
        for i in xrange(iterations):
            sample_qa = textData.get_random_qapair()
            q = sample_qa[0]
            a = sample_qa[1]
            print ">> "
            print q
            print "Real Response:"
            print a
            print "greedy:"
            print model.forward_greedy(q)
            print "beam search:"
            # for response in model.forward_beam(conversation_input):
            # logger.info(response.get_concat_sentence())
            # logger.info(model.response(response.get_concat_sentence()))
            print model.response(model.forward_beam(q)[0].get_concat_sentence())
            print "sample:"
            print model.forward_sample(q)
            print "\n"