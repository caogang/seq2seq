import mxnet as mx
import numpy as np
import data_utils
from bucket_io import BucketSentenceIter
from lstm import seq2seq_lstm_unroll
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

    devs = mx.context.gpu(1)

    # Put needed training stage here
    training_stage = [1]

    # Stage 1, Episode 1
    # Forward seq2seq

    if 1 in training_stage:
        def sym_gen(seq_len):
            return seq2seq_lstm_unroll(seq_len, num_hidden, num_embed,
                    num_vocab=textData.getVocabularySize(), num_layer = num_layers, dropout=args.dropout)
        forward_seq2seq_sym = sym_gen

        init_c = [("encode_init_c", (batch_size, num_layers, num_hidden))]
        init_h = [("encode_init_h", (batch_size, num_layers, num_hidden))]
        init_states = init_c + init_h

        Forward_data_feed = True
        forward_data_train = CornellDataIter(textData, [args.maxLength,], args.batchSize, init_states, True)
                                        #BucketSentenceIter(target_path, vocab,
                                        #buckets, batch_size,num_layers, init_states, Forward_data_feed)

        optimizer = mx.optimizer.SGD(momentum = momentum,
                                     learning_rate = learning_rate,
                                     clip_gradient = clip_norm)

        model = mx.model.FeedForward(ctx = devs,
                                     symbol = forward_seq2seq_sym,
                                     num_epoch = num_epoch,
                                     learning_rate = learning_rate,
                                     optimizer = optimizer,
                                     momentum = momentum,
                                     wd = 0,
                                     initializer = mx.initializer.Uniform(scale=0.07))
        model.fit(X = forward_data_train,
                  eval_metric = GroupAccuracy(),
                  batch_end_callback=mx.callback.Speedometer(batch_size, 50),
                  epoch_end_callback=mx.callback.do_checkpoint("../snapshots/seq2seq_newdata", period = 50))
