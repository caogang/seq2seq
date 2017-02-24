import re
import os
import sys
import logging


_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

file_termimal = "__file_terminal__\n"


BUCKETS = [5, 10, 20, 30, 40, 50, 60]

# def get_dialog_train_set_path(path):
#   return os.path.join(path, 'chat')


# def get_dialog_dev_set_path(path):
#   return os.path.join(path, 'chat_test')


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w.lower() for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True):
	if not os.path.exists(vocabulary_path):
		logging.debug("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
		vocab = {}
		with open(data_path, "r") as data_file:
			counter = 0
			for line in data_file:
				counter += 1
				if counter % 100000 == 0:
					logging.debug("Processing line %d" % counter)
				if line == file_termimal:
					continue
				tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
				for w in tokens:
					word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w 
					if word in vocab:
						vocab[word] += 1
					else:
						vocab[word] = 1
			vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
			if len(vocab_list) > max_vocabulary_size:
				vocab_list = vocab_list[:max_vocabulary_size]
			with open(vocabulary_path, "w") as write_file:
				for w in vocab_list:
					write_file.write(w + "\n")
		logging.info("Vocabulary created.")
	else:
		logging.info("Vocabulary exists. Skipping the step to create vocabulary.")


def initialize_vocabulary(vocabulary_path):
	if os.path.exists(vocabulary_path):
		rev_vocab = []
		with open(vocabulary_path, "r") as read_file:
			rev_vocab.extend(read_file.readlines())

		rev_vocab = [line.strip() for line in rev_vocab]
		vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])
		return vocab, rev_vocab
	else:
		raise ValueError("Vocabulary file %s not found", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):
	"""
	args:
	sentence: str
	vocabulary: dict
	tokenizer: function
	normalize_digits: bool
	"""
	if tokenizer:
		words = tokenizer(sentence)
	else:
		words = basic_tokenizer(sentence)

	if not normalize_digits:
		return [vocabulary.get(w, UNK_ID) for w in words]

	return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]	

def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None, normalize_digits=True):
	if not os.path.exists(target_path):
		logging.debug("Tokenizing data in %s" % data_path)
		vocab, _ = initialize_vocabulary(vocabulary_path)
		with open(data_path, "r") as data_file:
			with open(target_path, "w") as tokens_file:
				counter = 0
				for line in data_file:
					counter += 1
					if counter % 100000 == 0:
						print("Tokenizing line %d" % counter)
					if line == file_termimal:
						tokens_file.write(file_termimal)
						continue
					token_ids = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
					tokens_file.write(" ".join([str(token) for token in token_ids]) + "\n")
		logging.info("Finished Tokenize data in %s" % data_path)
	else:
		logging.info("There exists tokenized file in %s" % target_path)



if __name__ == "__main__":
	data_path = "../data/movie_lines_selected.txt"
	target_path = "../data/tokenized_data_movie_small.txt"
	vocabulary_path = "../data/vocabulary_movie_small.txt"
	max_vocabulary_size = 30000
	create_vocabulary(vocabulary_path, data_path, max_vocabulary_size)
	data_to_token_ids(data_path, target_path, vocabulary_path)