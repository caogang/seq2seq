import sys
import mxnet as mx


def hierarchyDiscriminatorSymbol(inputSeqLen, outputSeqLen, contentLen,
                                 inputHiddenNums, outputHiddenNums, contentHiddenNums,
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
    def __init__(self):

        pass
