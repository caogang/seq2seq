import sys
import mxnet as mx
sys.path.append('../')
sys.path.append('./')

from params import getArgs
from textdata import TextData
from seq2seq_model import Seq2SeqInferenceModelCornellData

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
    inputEncoderWeight = mx.sym.Variable('inputEncoderWeight')

    # Output Encoder
    outputEncoderInitC = mx.sym.Variable('outputEncoderInitC')
    outputEncoderInitC = mx.sym.Reshape(data=outputEncoderInitC, shape=(outputLayerNums, -1, outputHiddenNums))
    outputEncoderInitH = mx.sym.Variable('outputEncoderInitH')
    outputEncoderInitH = mx.sym.Reshape(data=outputEncoderInitH, shape=(outputLayerNums, -1, outputHiddenNums))
    outputEncoderWeight = mx.sym.Variable('outputEncoderWeight')

    # Content Encoder
    contentEncoderInitC = mx.sym.Variable('contentEncoderInitC')
    contentEncoderInitC = mx.sym.Reshape(data=contentEncoderInitC, shape=(contentLayerNums, -1, contentHiddenNums))
    contentEncoderInitH = mx.sym.Variable('contentEncoderInitH')
    contentEncoderInitH = mx.sym.Reshape(data=contentEncoderInitH, shape=(contentLayerNums, -1, contentHiddenNums))
    contentEncoderWeight = mx.sym.Variable('contentEncoderWeight')

    # Embedding Layer
    inputData = mx.sym.Variable('inputData')
    inputData = mx.sym.transpose(inputData)
    outputData = mx.sym.Variable('outputData')
    outputData = mx.sym.transpose(outputData)
    embedWeight = mx.sym.Variable('embedWeight')

    # Logistic Classifier
    clsWeight = mx.sym.Variable('clsWeight')
    clsBias = mx.sym.Variable('clsBias')
    label = mx.sym.Variable('softmaxLabel')

    # -----------Construct Symbols----------- #

    # Embedding Symbol
    inputEmbed = mx.sym.Embedding(data=inputData,
                                  input_dim=vocabNums,
                                  weight=embedWeight,
                                  output_dim=embedNums,
                                  name='inputEmbed')
    outputEmbed = mx.sym.Embedding(data=outputData,
                                   input_dim=vocabNums,
                                   weight=embedWeight,
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
    hInputEncoder = inputEncoder[1]
    cinputEncoder = inputEncoder[2]
    oOutputEncoder = outputEncoder[0]
    hOutputEncoder = outputEncoder[1]
    cOutputEncoder = outputEncoder[2]

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
    hContentEncoder = contentEncoder[1]
    cContentEncoder = contentEncoder[2]

    # 2-SoftmaxOut Symbol
    hContentEncoderReshape = mx.sym.Reshape(data=hContentEncoder, shape=(-1, contentHiddenNums))
    pred = mx.sym.FullyConnected(data=hContentEncoderReshape,
                                 num_hidden=2,
                                 weight=clsWeight,
                                 bias=clsBias,
                                 name='pred')
    binaryClassifier = mx.sym.SoftmaxOutput(data=pred,
                                            label=label,
                                            name='2-softmax',
                                            use_ignore=True)

    return mx.sym.Group([binaryClassifier,
                         mx.sym.BlockGrad(data=oInputEncoder),
                         mx.sym.BlockGrad(data=cinputEncoder),
                         mx.sym.BlockGrad(data=oOutputEncoder),
                         mx.sym.BlockGrad(data=cOutputEncoder),
                         mx.sym.BlockGrad(data=oContentEncoder),
                         mx.sym.BlockGrad(data=cContentEncoder)])


class HierarchyDiscriminatorModel:
    def __init__(self, args, text_data):
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

        self.momentum = 0.0
        self.clip_norm = 1.0
        self.learning_rate = args.learningRate
        self.num_epoch = args.numEpochs
        beam_size = 5  # 10

        args.maxLengthEnco = args.maxLength
        args.maxLengthDeco = args.maxLength + 2
        self.input_seq_len = args.maxLengthEnco
        self.output_seq_len = args.maxLengthDeco

        if args.load is None:
            args.load = 50

        self.devs = mx.context.gpu(0)

        _, arg_params, __ = mx.model.load_checkpoint("../snapshots/seq2seq_newdata", args.load)
        model = Seq2SeqInferenceModelCornellData(args.maxLength, 1, self.learning_rate,
                                                 text_data, args.hiddenSize, args.embeddingSize, args.numLayers,
                                                 arg_params, beam_size,
                                                 ctx=self.devs, dropout=0.)

        self.data = DiscriminatorData(args, text_data, model, forceRegenerate=False)
        pass

    def train(self):
        init_h = [('outputEncoderInitH', (self.batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitH', (self.batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitH', (self.batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_c = [('outputEncoderInitC', (self.batch_size, self.output_layer_nums, self.output_hidden_nums)),
                  ('contentEncoderInitC', (self.batch_size, self.content_layer_nums, self.content_hidden_nums)),
                  ('inputEncoderInitC', (self.batch_size, self.input_layer_nums, self.input_hidden_nums))]
        init_stats = init_c + init_h

        def sym_gen(seq_len):
            return hierarchyDiscriminatorSymbol(self.input_hidden_nums, self.output_hidden_nums, self.content_hidden_nums,
                                 self.input_layer_nums, self.output_layer_nums, self.content_layer_nums,
                                 self.embedding_size, self.vocab_nums, dropout=0.)

        data_train = DiscriminatorDataIter(self.data, self.batch_size, init_stats, self.input_seq_len, self.output_seq_len)

        optimizer = mx.optimizer.SGD(momentum = self.momentum,
                                     learning_rate = self.learning_rate,
                                     clip_gradient = self.clip_norm)

        model = mx.model.FeedForward(ctx = self.devs,
                                     symbol = hierarchyDiscriminatorSymbol,
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


if __name__ == '__main__':
    args = getArgs()
    origin_data = TextData(args)
    discriminator_model = HierarchyDiscriminatorModel(args, origin_data)
    discriminator_model.train()