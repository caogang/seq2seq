import mxnet as mx
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
from collections import namedtuple

def test4():
    data = mx.sym.Variable("data")
    h_init = mx.sym.Variable("h_init")
    c_init = mx.sym.Variable("c_init")
    param = mx.sym.Variable("param")
    state_size = 256
    num_layer = 1
    sym = mx.sym.RNN(data=data, parameters=param, state=h_init, state_cell=c_init, state_size=state_size,num_layers=num_layer,state_outputs=True,bidirectional=False, mode='lstm', name="lstmCell")
    input_shape = {"data":(10,1,256),"h_init":(1,1,256),"c_init":(1,1,256)}
    req = {"data":"null","param":"write","h_init":"null","c_init":"null"}
    model = sym.simple_bind(ctx = mx.gpu(0), grad_req = req, **input_shape)
    logger.debug(sym.list_outputs())
    logger.debug(sym.list_arguments())



if __name__ == "__main__":
    test4()

