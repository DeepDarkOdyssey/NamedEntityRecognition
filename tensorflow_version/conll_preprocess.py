"""Read, split and save the CoNLL 2002 dataset for our model"""

import csv
import os
import random
import argparse
import json


def load_data(csv_path):
    """Loads the dataset into memory"""
    with open(csv_path, encoding='windows-1252') as f:
        csv_file = csv.reader(f, delimiter=',')
        words, labels = [], []
        data = []

        for r_idx, row in enumerate(csv_file):
            if r_idx == 0:
                continue
            sentence_flag, word, pos, label = row
            if sentence_flag != '':
                assert len(words) == len(labels)
                data.append((' '.join(words), ' '.join(labels)))
                words, labels = [], []

            try:
                words.append(word)
                labels.append(label)
            except UnicodeDecodeError as e:
                print('An Exception was raised {}'.format(e))
                pass

    return data


def save_data(data, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    :param data: list of tuples like (sentence, label)
    :param save_dir: string
    """
    print('Saving data to {}...'.format(save_dir), end=' ')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'sentences.txt'), 'w', encoding='utf8') as sentence_file:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as label_file:
            for sentence, label in data:
                sentence_file.write(sentence + '\n')
                label_file.write(label + '\n')
    print('- Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process raw data and split it into train, dev, test set')
    parser.add_argument('--raw_data_path', type=str, default='data/CoNLL2002/ner_dataset.csv',
                        help='specify the raw data you want to load and process')
    parser.add_argument('--save_dir', type=str, default='data/CoNLL2002',
                        help='specify the directory you want to store the processed and split data')
    args = parser.parse_args()

    # load data from original csv
    data = load_data(args.raw_data_path)

    # random shuffle the data and split it into train, dev and test
    random.shuffle(data)
    train_data = data[: int(0.8*len(data))]
    dev_data = data[int(0.8*len(data)): int(0.9*len(data))]
    test_data = data[int(0.9*len(data)):]

    # save the split data to files
    save_data(train_data, os.path.join(args.save_dir, 'train'))
    save_data(dev_data, os.path.join(args.save_dir, 'dev'))
    save_data(test_data, os.path.join(args.save_dir, 'test'))

    # save global config
    global_config = {
        'train_size': len(train_data),
        'dev_size': len(dev_data),
        'test_size': len(test_data)
    }
    with open('global_config.json', 'w') as f:
        json.dump(global_config, f, indent=4)
    print('Saving global config to {}'.format(os.path.join(os.path.abspath('.'), 'global_config.json')))