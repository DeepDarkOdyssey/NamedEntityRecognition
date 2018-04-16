import json
import os
import random

import tensorflow as tf

from conll_preprocess import load_data, save_data
from input_fn import build_dataset, build_inputs
from models import build_model
from utils import load_config, set_logger, prepare_dirs
from vocab import Vocab

base_dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

# experiment configuration
flags.DEFINE_string('model_name', 'BaselineModel', 'Unique name of the model and should exists in models/__init__.py')
flags.DEFINE_string('experiments_dir', base_dir + '/experiments', 'Directory to store your experiments')
flags.DEFINE_string('exp_name', 'test', 'Unique name for this experiment')
flags.DEFINE_string('global_config', base_dir + '/global_config.json', 'Path to the global config')
flags.DEFINE_string('config_path', '', 'Specify a path to load config that has been saved before')
flags.DEFINE_bool('prepare', False, 'Process the raw data and prepare the vocab')
flags.DEFINE_string('raw_data', base_dir + '/data/CoNLL2002/ner_dataset.csv', 'Path to the raw data file')
flags.DEFINE_string('data_dir', base_dir + '/data/CoNLL2002', 'Directory to save the processed an split data')
flags.DEFINE_boolean('train', False, 'Train the model with current config and store in the specified experiment dir')
# flags.DEFINE_string('exp_dir', os.path.join(flags.FLAGS.experiments_dir, flags.FLAGS.model_name, flags.FLAGS.exp_name),
#                     'The directory to store all the things of this experiment')
flags.DEFINE_string('vocab_dir', base_dir + '/data/CoNLL2002/vocab', 'Directory to save the vocab')
flags.DEFINE_string('logger_name', 'test', "Logger's name")
flags.DEFINE_string('predict', '', 'Predict on the test set and save the results')

# model hyperparameters
flags.DEFINE_integer('embed_size', 50, 'The embedding size for words')
flags.DEFINE_integer('hidden_size', 50, 'Number of hidden units in each layer')
flags.DEFINE_string('rnn_mode', 'uni', 'Unfold rnn in one direction or both directions, can be either "uni" or "bi"')
flags.DEFINE_string('cell_type', 'gru', 'Type of the rnn cell, can be "rnn" or "lstm" or "gru"')
flags.DEFINE_integer('num_layers', 1, 'How many rnn layers you want to stack')
flags.DEFINE_float('learning_rate', 0.001, 'The learning rate of the optimizer')
flags.DEFINE_float('keep_prob', 1.0, 'The dropout keep probability')

# train settings
flags.DEFINE_integer('batch_size', 50, 'The batch size of datasets')
flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs')
flags.DEFINE_string('restore_from', '', 'Path of directory of weights you want your model to restore from')
flags.DEFINE_integer('log_freq', 50, 'The frequency of logging')
flags.DEFINE_integer('summary_freq', 50, 'The frequency of saving summaries')


def main(_):
    config = load_config(flags.FLAGS.flag_values_dict())
    print(config)

    if config.prepare:
        # load data from original csv
        data = load_data(config.raw_data)

        # random shuffle the data and split it into train, dev and test
        random.shuffle(data)
        train_data = data[: int(0.8*len(data))]
        dev_data = data[int(0.8*len(data)): int(0.9*len(data))]
        test_data = data[int(0.9*len(data)):]

        # save the split data to files
        save_data(train_data, os.path.join(config.data_dir, 'train'))
        save_data(dev_data, os.path.join(config.data_dir, 'dev'))
        save_data(test_data, os.path.join(config.data_dir, 'test'))

        # prepare vocab
        word_vocab = Vocab(fins=[os.path.join(config.data_dir, 'train', 'sentences.txt')])
        label_vocab = Vocab(fins=[os.path.join(config.data_dir, 'train', 'labels.txt')])
        if not os.path.exists(config.vocab_dir):
            os.mkdir(config.vocab_dir)
        word_vocab.save_to(os.path.join(config.vocab_dir, 'word_vocab.pkl'))
        label_vocab.save_to(os.path.join(config.vocab_dir, 'label_vocab.pkl'))
        print('Saving vocab to {}'.format(config.vocab_dir))

        # save global config
        global_config = {
            'train_size': len(train_data),
            'dev_size': len(dev_data),
            'test_size': len(test_data),
            'word_vocab_size': word_vocab.size(),
            'label_vocab_size': label_vocab.size()
        }
        with open(config.global_config, 'w') as f:
            json.dump(global_config, f, indent=4)
        print('Saving global config to {}'.format(config.global_config))

    if config.train:
        # modify the config
        if config.global_config:
            config = load_config(config, config.global_config)
        if config.config_path:
            config = load_config(config, config.config_path)

        # prepare dirs
        config = prepare_dirs(config)

        # save current config to the experiment directory
        with open(os.path.join(config.exp_dir, 'config.json'), 'w') as f:
            json.dump(config._asdict(), f, indent=4)

        # set logger
        set_logger(config)

        # load vocab
        word_vocab = Vocab()
        word_vocab.load_from(os.path.join(config.vocab_dir, 'word_vocab.pkl'))
        label_vocab = Vocab()
        label_vocab.load_from(os.path.join(config.vocab_dir, 'label_vocab.pkl'))

        # load dataset
        train_set = build_dataset(config, 'train', word_vocab, label_vocab)
        dev_set = build_dataset(config, 'dev', word_vocab, label_vocab)

        # build inputs
        inputs = build_inputs(train_set.output_types, train_set.output_shapes)

        # build model
        # model = BaselineModel(config, inputs, word_vocab, label_vocab)
        model = build_model(config, inputs, word_vocab, label_vocab)

        # train the model
        model.train(train_set, dev_set)

    if config.predict:
        # TODO: finish the predict part
        pass


if __name__ == '__main__':
    tf.app.run()

