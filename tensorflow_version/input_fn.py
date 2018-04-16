"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import tensorflow.contrib as tc
import os
from collections import namedtuple

BatchInput = namedtuple('BatchInput',
                        ['iterator', 'word_ids', 'label_ids', 'sentence_lengths'])


def build_dataset(config, mode, word_vocab, label_vocab, num_parallel_calls=8):
    with tf.device('/cpu:0'):
        # set up look up tables for dataset
        word_table = tc.lookup.index_table_from_tensor(word_vocab.id2token,
                                                       default_value=word_vocab.token2id[word_vocab.unk_token])
        label_table = tc.lookup.index_table_from_tensor(label_vocab.id2token,
                                                        default_value=label_vocab.token2id[label_vocab.unk_token])

        # read data from files
        sentences = tf.data.TextLineDataset(os.path.join(config.data_dir, mode, 'sentences.txt'))
        sentences = sentences.map(lambda string: tf.string_split([string]).values,
                                  num_parallel_calls=num_parallel_calls)
        sentences = sentences.map(lambda tokens: (tf.to_int32(word_table.lookup(tokens)), tf.size(tokens)),
                                  num_parallel_calls=num_parallel_calls)

        labels = tf.data.TextLineDataset(os.path.join(config.data_dir, mode, 'labels.txt'))
        labels = labels.map(lambda string: tf.string_split([string]).values,
                            num_parallel_calls=num_parallel_calls)
        labels = labels.map(lambda tokens: (tf.to_int32(label_table.lookup(tokens)), tf.size(tokens)),
                            num_parallel_calls=num_parallel_calls)

        # zip the sentences and labels together
        dataset = tf.data.Dataset.zip((sentences, labels))

        # pad and batch
        padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])),
                         (tf.TensorShape([None]), tf.TensorShape([])))
        padding_values = ((word_vocab.token2id[word_vocab.pad_token], 0),
                          (label_vocab.token2id[label_vocab.pad_token], 0))
        dataset = dataset.padded_batch(config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

        # prefetch 1 to accelerate the input following guide from
        # https://www.tensorflow.org/performance/datasets_performance
        dataset = dataset.prefetch(1)

        return dataset


def build_inputs(output_types, output_shapes):
    # build an iterator with specific output types and shapes that can switch the underline dataset
    iterator = tf.data.Iterator.from_structure(output_types, output_shapes)
    ((word_ids, sentence_lengths), (label_ids, _)) = iterator.get_next()
    return BatchInput(iterator, word_ids, label_ids, sentence_lengths)

