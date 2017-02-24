## An instruction of the script

### Introduction


### Functionality of each file

* bucket_io.py  -> Generate a data iterator with bucket for seq2seq model, autofeed for each batch and auto shuffle for each epoch
* data_utils.py  -> Read the files, generate vocabulary, convert the document into word ids etc. 
* lstm.py  -> It contains all the rnn-lstm net structure including lstm_unroll, encoding_inference, decoding_inference, unroll_without softmax(to allow the calculated gradient passing into the net), softmax(to calculate the gradient)
* mutual_information_model.py  -> An high level class module, second stage of the RL training. It has two input sentence as encoding data, then gets reponses in beam size. It calculates the maximum mutual information score (MMI) with the input and generated reponse to find the gradient. Then pass the gradient back to update_model to train. It loads params from update_model at the beginning of each epoch. Refer to http://arxiv.org/abs/1606.01541 for more detail.
* predict.py  -> It directly uses inference model to generate the responses to check the answer by human intuition.
* seq2seq_model.py  -> It is a inference model with greedy search and beam search api.
* train.py  -> It divide the training into three stage. First stage is for forward seq2seq learning and backward seq2seq learning. Second stage is for mutual information reinforcement learning training. Third stage is for dialogue simulation reinforcement learning training.

### How to use it.

* Put desired data into data directory
* Change the data_path in train.py
* Change the desired stage number in train.py 
* Change the desired parameter path that you want
* >> python train.py


### How to get the data

* Find opensubstitle zip file in yichengong@gpu002.horizon-robotics.com:/home/yichengong/projects/chatbot/data
* Run the code in extract_opensubtitle_xml.ipynb to extract the content of opensubtitle completely
  * It will add __file__terminal__ at the end of each file
  * The script assume that the file is unziped. So unzip it first or you can add something to show your intellectual superiority

### Attention

* You might want to copy the files from test dir to src dir, in case you want to run them.