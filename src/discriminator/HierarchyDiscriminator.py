import sys
import mxnet as mx
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
    hInputEncoder = mx.sym.Select(*[oInputEncoder, inputEncoder[1], cInputEncoder], index=1)

    oOutputEncoder = outputEncoder[0]
    cOutputEncoder = outputEncoder[2]
    hOutputEncoder = mx.sym.Select(*[oOutputEncoder, outputEncoder[1], cOutputEncoder], index=1)

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
    hContentEncoder = mx.sym.Select(*[oContentEncoder, contentEncoder[1], cContentEncoder], index=1)

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

    return binaryClassifier


class HierarchyDiscriminatorModel:
    def __init__(self, args, text_data, is_train=True):
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
        print self.num_epoch
        beam_size = 5  # 10

        args.maxLengthEnco = args.maxLength
        args.maxLengthDeco = args.maxLength + 2
        self.input_seq_len = args.maxLengthEnco
        self.output_seq_len = args.maxLengthDeco

        self.devs = mx.context.gpu(0)

        if args.load is None:
            args.load = 50

        _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
        model = Seq2SeqInferenceModelCornellData(args.maxLength, 1, self.learning_rate,
                                                 text_data, args.hiddenSize, args.embeddingSize, args.numLayers,
                                                 arg_params, beam_size,
                                                 ctx=self.devs, dropout=0.)

        self.data = DiscriminatorData(args, text_data, model, forceRegenerate=False)

        self.is_train = is_train
        if not is_train:
            test_sym, dis_arg_params, dis_aux_params = mx.model.load_checkpoint("../snapshots/discriminator", args.loadDis)

            sym = hierarchyDiscriminatorSymbol(self.input_hidden_nums, self.output_hidden_nums,
                                                self.content_hidden_nums,
                                                self.input_layer_nums, self.output_layer_nums, self.content_layer_nums,
                                                self.embedding_size, self.vocab_nums, dropout=0.)

            # print test_sym.list_arguments()
            # print sym.list_arguments()
            # print dis_arg_params

            self.pretrained_model = mx.mod.Module(sym, context=self.devs)

            batch_size = 1

            init_h = [('outputEncoderInitH', (batch_size, self.output_layer_nums, self.output_hidden_nums)),
                      ('contentEncoderInitH', (batch_size, self.content_layer_nums, self.content_hidden_nums)),
                      ('inputEncoderInitH', (batch_size, self.input_layer_nums, self.input_hidden_nums))]
            init_c = [('outputEncoderInitC', (batch_size, self.output_layer_nums, self.output_hidden_nums)),
                      ('contentEncoderInitC', (batch_size, self.content_layer_nums, self.content_hidden_nums)),
                      ('inputEncoderInitC', (batch_size, self.input_layer_nums, self.input_hidden_nums))]
            self.init_stats = init_c + init_h

            provide_data = [('inputData', (batch_size, self.input_seq_len)),
                                 ('outputData', (batch_size, self.input_seq_len))] + self.init_stats
            provide_label = ['softmaxLabel']
            # print provide_data
            # print provide_label
            self.pretrained_model.bind(data_shapes=provide_data,
                            #label_shapes=provide_label,
                            for_training=False)
            # self.pretrained_model.init_params()
            self.pretrained_model.set_params(arg_params=dis_arg_params, aux_params=dis_aux_params, allow_missing=True)
        pass

    def train(self):
        init_h = [('outputEncoderInitH', (self.batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitH', (self.batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitH', (self.batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_c = [('outputEncoderInitC', (self.batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitC', (self.batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitC', (self.batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_stats = init_c + init_h

        data_train = DiscriminatorDataIter(self.data, self.batch_size, init_stats, self.input_seq_len, self.input_seq_len)

        optimizer = mx.optimizer.SGD(momentum = self.momentum,
                                     learning_rate = self.learning_rate,
                                     clip_gradient = self.clip_norm)

        def sym_gen(seq_len):
            return hierarchyDiscriminatorSymbol(self.input_hidden_nums, self.output_hidden_nums,
                                                self.content_hidden_nums,
                                                self.input_layer_nums, self.output_layer_nums, self.content_layer_nums,
                                                self.embedding_size, self.vocab_nums, dropout=0.)

        model = mx.model.FeedForward(ctx = self.devs,
                                     symbol = sym_gen,
                                     num_epoch = self.num_epoch,
                                     learning_rate = self.learning_rate,
                                     optimizer = optimizer,
                                     momentum = self.momentum,
                                     wd = 0,
                                     initializer = mx.initializer.Uniform(scale=0.07))
        model.fit(X = data_train,
                  eval_metric = "accuracy",
                  batch_end_callback=mx.callback.Speedometer(self.batch_size, 50),
                  epoch_end_callback=mx.callback.do_checkpoint("../snapshots/discriminator", period = 50))
        pass

    def predict(self, q, a):
        batch = self.generate_batch(q, a)
        self.pretrained_model.forward(batch)
        prob_list = self.pretrained_model.get_outputs()[0].asnumpy()
        print prob_list
        pass

    def generate_batch(self, q, a):
        q = q.rstrip('<eos>')
        a = a.rstrip('<eos>')
        q_id = self.text_data.sentence2id(q)
        a_id = self.text_data.sentence2id(a)

        q_padded = [self.text_data.padToken] * (self.args.maxLengthEnco - len(q_id)) + q_id  # Left padding for the input
        a_padded = [self.text_data.padToken] * (self.args.maxLengthEnco - len(a_id)) + a_id

        init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_stats]
        init_state_names = [x[0] for x in self.init_stats]
        batch_input_seq = mx.nd.array(q_padded)
        batch_output_seq = mx.nd.array(a_padded)
        batch_input_seq = batch_input_seq.reshape((1, self.args.maxLengthEnco))
        batch_output_seq = batch_output_seq.reshape((1, self.args.maxLengthEnco))
        # print batch_input_seq.shape, batch_output_seq.shape
        data_all = [batch_input_seq, batch_output_seq] + init_state_arrays
        data_names = ["inputData", "outputData"] + init_state_names
        data_batch = SimpleDiscriminatorBatch(data_names, data_all, [], [], self.args.maxLengthEnco)
        return data_batch


if __name__ == '__main__':
    args = getArgs()
    origin_data = TextData(args)
    discriminator_model = HierarchyDiscriminatorModel(args, origin_data)
    discriminator_model.train()
    discriminator_inference_model = HierarchyDiscriminatorModel(args, origin_data, is_train=False)
    discriminator_inference_model.predict("hi . <eos>", "hello . <eos>")
    discriminator_inference_model.predict("who are you ? <eos>", "i 'm bob . <eos>")
