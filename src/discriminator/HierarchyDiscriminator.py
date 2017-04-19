import sys
import mxnet as mx
import numpy as np
sys.path.append('../')
sys.path.append('./')
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from params import getArgs
from textdata import TextData
from seq2seq_model import Seq2SeqInferenceModelCornellData
from DiscriminatorDataIter import SimpleDiscriminatorBatch

from DiscriminatorDataIter import DiscriminatorDataIter
from DiscriminatorDataGenerator import DiscriminatorData

class GroupAccuracy(mx.metric.EvalMetric):
    """Calculate group accuracy."""

    def __init__(self):
        super(GroupAccuracy, self).__init__('group accuracy')

    def update(self, labels, preds):
        preds = [preds[0]]
        mx.metric.check_label_shapes(labels, preds)
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax_channel(pred_label)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            #print label, pred_label
            mx.metric.check_label_shapes(label, pred_label)
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

def hierarchyDiscriminatorSymbol(inputHiddenNums, outputHiddenNums, contentHiddenNums,
                                 inputLayerNums, outputLayerNums, contentLayerNums,
                                 embedNums, vocabNums, dropout=0.):
    # -----------Build Variables----------- #

    # Input Encoder
    inputEncoderInitC = mx.sym.Variable('inputEncoderInitC')
    inputEncoderInitC = mx.sym.Reshape(data=inputEncoderInitC, shape=(inputLayerNums, -1, inputHiddenNums))
    inputEncoderInitH = mx.sym.Variable('inputEncoderInitH')
    inputEncoderInitH = mx.sym.Reshape(data=inputEncoderInitH, shape=(inputLayerNums, -1, inputHiddenNums))
    inputEncoderWeight = mx.sym.Variable('input_encoder_weight')

    # Output Encoder
    outputEncoderInitC = mx.sym.Variable('outputEncoderInitC')
    outputEncoderInitC = mx.sym.Reshape(data=outputEncoderInitC, shape=(outputLayerNums, -1, outputHiddenNums))
    outputEncoderInitH = mx.sym.Variable('outputEncoderInitH')
    outputEncoderInitH = mx.sym.Reshape(data=outputEncoderInitH, shape=(outputLayerNums, -1, outputHiddenNums))
    outputEncoderWeight = mx.sym.Variable('output_encoder_weight')

    # Content Encoder
    contentEncoderInitC = mx.sym.Variable('contentEncoderInitC')
    contentEncoderInitC = mx.sym.Reshape(data=contentEncoderInitC, shape=(contentLayerNums, -1, contentHiddenNums))
    contentEncoderInitH = mx.sym.Variable('contentEncoderInitH')
    contentEncoderInitH = mx.sym.Reshape(data=contentEncoderInitH, shape=(contentLayerNums, -1, contentHiddenNums))
    contentEncoderWeight = mx.sym.Variable('content_encoder_weight')

    # Embedding Layer
    inputData = mx.sym.Variable('inputData')
    inputData = mx.sym.transpose(inputData)
    outputData = mx.sym.Variable('outputData')
    outputData = mx.sym.transpose(outputData)
    # embedWeight = mx.sym.Variable('embed_weight')

    # Logistic Classifier
    # clsWeight = mx.sym.Variable('cls_weight')
    # clsBias = mx.sym.Variable('cls_bias')
    label = mx.sym.Variable('softmaxLabel')

    # -----------Construct Symbols----------- #

    # Embedding Symbol
    inputEmbed = mx.sym.Embedding(data=inputData,
                                  input_dim=vocabNums,
                                  # weight=embedWeight,
                                  output_dim=embedNums,
                                  name='inputEmbed')
    outputEmbed = mx.sym.Embedding(data=outputData,
                                   input_dim=vocabNums,
                                   # weight=embedWeight,
                                   output_dim=embedNums,
                                   name='outputEmbed')

    # Encoder Symbol
    inputEncoder = mx.sym.RNN(data=inputEmbed,
                              parameters=inputEncoderWeight,
                              state=inputEncoderInitH,
                              state_cell=inputEncoderInitC,
                              state_size=inputHiddenNums,
                              num_layers=inputLayerNums,
                              state_outputs=True,
                              mode='lstm',
                              name='inputEncoder')
    outputEncoder = mx.sym.RNN(data=outputEmbed,
                               parameters=outputEncoderWeight,
                               state=outputEncoderInitH,
                               state_cell=outputEncoderInitC,
                               state_size=outputHiddenNums,
                               num_layers=outputLayerNums,
                               state_outputs=True,
                               mode='lstm',
                               name='outputEncoder')

    oInputEncoder = inputEncoder[0]
    cInputEncoder = inputEncoder[2]
    hInputEncoder = inputEncoder[1]#mx.sym.Select(*[oInputEncoder, inputEncoder[1], cInputEncoder], index=1)

    oOutputEncoder = outputEncoder[0]
    cOutputEncoder = outputEncoder[2]
    hOutputEncoder = outputEncoder[1]#mx.sym.Select(*[oOutputEncoder, outputEncoder[1], cOutputEncoder], index=1)

    # Concat content data from hInputEncoder and hOutputEncoder
    contentData = mx.sym.Concat(hInputEncoder, hOutputEncoder, dim=0)

    # Content Encoder Symbol
    contentEncoder = mx.sym.RNN(data=contentData,
                                parameters=contentEncoderWeight,
                                state=contentEncoderInitH,
                                state_cell=contentEncoderInitC,
                                state_size=contentHiddenNums,
                                num_layers=contentLayerNums,
                                state_outputs=True,
                                mode='lstm',
                                name='contentEncoder')

    oContentEncoder = contentEncoder[0]
    cContentEncoder = contentEncoder[2]
    hContentEncoder = contentEncoder[1]#mx.sym.Select(*[oContentEncoder, contentEncoder[1], cContentEncoder], index=1)

    # 2-SoftmaxOut Symbol
    hContentEncoderReshape = mx.sym.Reshape(data=hContentEncoder, shape=(-1, contentHiddenNums))
    pred = mx.sym.FullyConnected(data=hContentEncoderReshape,
                                 num_hidden=2,
                                 # weight=clsWeight,
                                 # bias=clsBias,
                                 name='pred')
    binaryClassifier = mx.sym.SoftmaxOutput(data=pred,
                                            label=label,
                                            name='2-softmax',
                                            use_ignore=True)

    return mx.sym.Group([binaryClassifier,
                         mx.sym.BlockGrad(data=oInputEncoder),
                         mx.sym.BlockGrad(data=oOutputEncoder),
                         mx.sym.BlockGrad(data=oContentEncoder),
                         mx.sym.BlockGrad(data=cInputEncoder),
                         mx.sym.BlockGrad(data=cOutputEncoder),
                         mx.sym.BlockGrad(data=cContentEncoder)])


class HierarchyDiscriminatorModel:
    def __init__(self, args, text_data, ctx=mx.context.gpu(0), is_train=True, prefix="../snapshots/discriminator"):
        self.args = args

        self.batch_size = args.batchSize
        self.input_layer_nums = args.inputLayerNums
        self.output_layer_nums = args.outputLayerNums
        self.input_hidden_nums = args.inputHiddenNums
        self.output_hidden_nums = args.outputHiddenNums
        self.content_layer_nums = args.contentLayerNums
        self.content_hidden_nums = args.contentHiddenNums
        self.embedding_size = args.embeddingSize
        self.vocab_nums = text_data.getVocabularySize()
        self.text_data = text_data

        self.momentum = 0.0
        self.clip_norm = 1.0
        self.learning_rate = args.learningRate
        self.num_epoch = args.numEpochDis
        self.prefix = prefix
        beam_size = 5  # 10

        args.maxLengthEnco = args.maxLength
        args.maxLengthDeco = args.maxLength + 2
        self.input_seq_len = args.maxLengthEnco
        self.output_seq_len = args.maxLengthDeco

        self.devs = ctx

        if args.load is None:
            args.load = 50

        _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
        model = Seq2SeqInferenceModelCornellData(args.maxLength, 1, self.learning_rate,
                                                 text_data, args.hiddenSize, args.embeddingSize, args.numLayers,
                                                 arg_params, beam_size,
                                                 ctx=self.devs, dropout=0.)

        self.data = DiscriminatorData(args, text_data, model, forceRegenerate=False)
        #self.data = DiscriminatorData(args, text_data, None, forceRegenerate=False)

        self.dis_arg_params, self.dis_aux_params = self.load_check_points(prefix)
        self.train_model = self.generate_model(self.batch_size, self.dis_arg_params, self.dis_aux_params)
        self.train_one_batch_model = self.generate_model(1, self.dis_arg_params, self.dis_aux_params)
        self.predict_model = self.generate_model(1, self.dis_arg_params, self.dis_aux_params, is_train=False)

        self.is_train = is_train
        pass

    def generate_model(self, batch_size, arg_params, aux_params, is_train=True):

        sym = hierarchyDiscriminatorSymbol(self.input_hidden_nums, self.output_hidden_nums,
                                           self.content_hidden_nums,
                                           self.input_layer_nums, self.output_layer_nums, self.content_layer_nums,
                                           self.embedding_size, self.vocab_nums, dropout=0.)

        # print test_sym.list_arguments()
        # print sym.list_arguments()
        # print dis_arg_params
        # sym.save('./dis.json')

        init_h = [('outputEncoderInitH', (batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitH', (batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitH', (batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_c = [('outputEncoderInitC', (batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitC', (batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitC', (batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_stats = init_c + init_h

        provide_data = [('inputData', (batch_size, self.input_seq_len)),
                        ('outputData', (batch_size, self.input_seq_len))] + init_stats
        data_names = []
        for d in provide_data:
            data_names.append(d[0])
        if is_train:
            model = mx.mod.Module(sym, data_names=data_names, label_names=['softmaxLabel'], context=self.devs)
        else:
            model = mx.mod.Module(sym, data_names=data_names, label_names=(), context=self.devs)
        # print provide_data
        # print provide_label
        if is_train:
            provide_label = [('softmaxLabel', (batch_size,))]
            model.bind(data_shapes=provide_data,
                       label_shapes=provide_label,
                       for_training=True)
            print 'after bind batch_size:' + str(batch_size) 
            if arg_params is None:
                model.init_params()
            else:
                model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
            model.init_optimizer(
                    optimizer='adam',
                    optimizer_params={
                        'learning_rate': 1e-4,
                        'beta1': 0.5,
                        'beta2': 0.9
                    })
        else:
            model.bind(data_shapes=provide_data,
                       for_training=False)
            if arg_params is not None:
                model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
        return model

    def train(self):
        self.train_model.set_params(arg_params=self.dis_arg_params, aux_params=self.dis_aux_params, allow_missing=True)

        init_h = [('outputEncoderInitH', (self.batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitH', (self.batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitH', (self.batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_c = [('outputEncoderInitC', (self.batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitC', (self.batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitC', (self.batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_stats = init_c + init_h
        data_train = DiscriminatorDataIter(self.data, self.batch_size, init_stats, self.input_seq_len, self.input_seq_len)
        self.train_model.fit(data_train,
                             eval_metric = GroupAccuracy(),
                             num_epoch=self.num_epoch,
                             batch_end_callback=mx.callback.Speedometer(self.batch_size, 50),
                             epoch_end_callback=mx.callback.do_checkpoint(self.prefix,
                                                                          period=100))
        self.dis_arg_params, self.dis_aux_params = self.train_model.get_params()

    def predict(self, q, a):
        self.predict_model.set_params(arg_params=self.dis_arg_params, aux_params=self.dis_aux_params, allow_missing=True)

        batch = self.generate_batch(q, a)
        self.predict_model.forward(batch)
        prob_list = self.predict_model.get_outputs()[0].asnumpy()
        prob_list = np.squeeze(prob_list)
        print prob_list
        return prob_list

    def train_one_batch(self, batch_tuple):
        self.train_one_batch_model.set_params(arg_params=self.dis_arg_params, aux_params=self.dis_aux_params, allow_missing=True)

        batch = self.generate_batch(batch_tuple[0],
                                    batch_tuple[1],
                                    label=batch_tuple[2],
                                    is_train=True)
        self.train_one_batch_model.forward(batch)
        self.train_one_batch_model.backward()
        self.train_one_batch_model.update()

        self.dis_arg_params, self.dis_aux_params = self.train_one_batch_model.get_params()

    def load_check_points(self, prefix):
        test_sym, dis_arg_params, dis_aux_params = mx.model.load_checkpoint(prefix, self.args.loadDis)
        return dis_arg_params, dis_aux_params

    def save_check_points(self, save_path, num_epoch):
        mx.model.save_checkpoint(save_path, num_epoch, self.train_model.symbol(),
                                 self.dis_arg_params, self.dis_aux_params)

    def generate_batch(self, q, a, label=None, is_train=False):
        q = q.rstrip('<eos>')
        a = a.rstrip('<eos>')
        print q + ' | ' + a
        q_id = self.text_data.sentence2id(q)
        a_id = self.text_data.sentence2id(a)
        print q_id, a_id

        q_padded = [self.text_data.padToken] * (self.args.maxLengthEnco - len(q_id)) + q_id  # Left padding for the input
        a_padded = [self.text_data.padToken] * (self.args.maxLengthEnco - len(a_id)) + a_id

        batch_size = 1

        init_h = [('outputEncoderInitH', (batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitH', (batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitH', (batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_c = [('outputEncoderInitC', (batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitC', (batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitC', (batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_stats = init_c + init_h

        init_state_arrays = [mx.nd.zeros(x[1]) for x in init_stats]
        init_state_names = [x[0] for x in init_stats]
        batch_input_seq = mx.nd.array(q_padded)
        batch_output_seq = mx.nd.array(a_padded)
        batch_input_seq = batch_input_seq.reshape((1, self.args.maxLengthEnco))
        batch_output_seq = batch_output_seq.reshape((1, self.args.maxLengthEnco))
        # print batch_input_seq.shape, batch_output_seq.shape
        data_all = [batch_input_seq, batch_output_seq] + init_state_arrays
        data_names = ["inputData", "outputData"] + init_state_names
        if is_train and label is not None:
            label_names = ["softmaxLabel"]
            label_all = mx.nd.array([label])
            data_batch = SimpleDiscriminatorBatch(data_names, data_all, label_names, label_all, self.args.maxLengthEnco)
            return data_batch
        else:
            data_batch = SimpleDiscriminatorBatch(data_names, data_all, [], [], self.args.maxLengthEnco)
            return data_batch


if __name__ == '__main__':
    args = getArgs()
    origin_data = TextData(args)
    prefix = "../snapshots/discriminator-new-optimizer"
    #discriminator_model = HierarchyDiscriminatorModel(args, origin_data, prefix=prefix)
    #discriminator_model.train()
    discriminator_inference_model = HierarchyDiscriminatorModel(args, origin_data, prefix=prefix)
    discriminator_inference_model.predict("hi . <eos>", "hello . <eos>")
    discriminator_inference_model.predict("hi . <eos>", "how are you . <eos>")
    discriminator_inference_model.predict("hi . <eos>", "i 'm fine . <eos>")
    discriminator_inference_model.predict("hi . <eos>", "what are you doing ? <eos>")
    discriminator_inference_model.predict("who are you ? <eos>", "i 'm bob . <eos>")
    discriminator_inference_model.predict("where is your pen ? <eos>", "i 'm bob . <eos>")
    discriminator_inference_model.predict("where is your pen ? <eos>", "in my bag . <eos>")
    discriminator_inference_model.predict("i like you . <eos>", "i like you too . <eos>")
    discriminator_inference_model.train_one_batch(("hi . <eos>", "hello . <eos>", 1))
    discriminator_inference_model.train_one_batch(("hi . <eos>", "fuck you . <eos>", 0))
    print 'train one batch finish'
    discriminator_inference_model.train()
    print 'finish'

