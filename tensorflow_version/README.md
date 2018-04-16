# Named Entity Recognition with Tensorflow

This project is inspired by the [CS230-project], built with personalized folder structure and additional models.

## Requirements
```
python3 
tensorflow >= 1.6
tqdm
```


## Task
Given a sentence, give a tag to each word ([Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition))

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```

## Demo Data
Following the [CS230-project], the project use the data from [Kaggel](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data), 
download the ner_dataset.csv and save it under the data directory(all wherever you'd like to save it, just remember the path).

## Usage
This contains several steps:
1. Before you can get started on training the model, you need to split the raw data in to train/dev/test sets and save
them in some easy-to-load format for the the tf.data APIs. Additionally, most nlp projects need some vocabularies and 
maybe the embedding for each token, This should be done together while processing the raw data. So, run the follow script
to prepare the data once for all.
```
python  run.py --prepare
```
> For other configurations like where to load and save the data and vocab and so on, check out the `run.py` for details, 
or try `python run.py -h`.

After running the prepare command, you shall find a `global_config.json` saved in current directory and some new directories 
and files has been created in the `data` directory.

2. After the dirty preprocessing jobs, you can try running an experiment with some configurations by:
```
python run.py --train model_name BaselineModel --exp_name test
```
This command will run a experiment with the name *test* using the *BaselineModel* defined in the `models` directory with
default configurations. You can modify the model by changing some hyperparameters if you want.

> Note that the all available model names are defined in `models/__init__.py`. If you build a model yourself, you
must add the name to the file mentioned above.

All the data related to this experiment will be saved in `experiments/BaselineModel/test`, including the checkpoints, 
summaries, logs and the current config, so you can check the performance of the model or reuse some config conveniently 
later.

3. When the training process in Done, you can run the following script to make predictions on some testset
```
python run.py --predict
```

## Folder Structure
The architecture of this project is based on [cs230-example](https://github.com/cs230-stanford/cs230-code-examples), 
[Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) and 
[this blog](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3).
```
├── data                        - this folder contains all the data files, the raw data, the train/dev/test sets and the vocab 
│   └── CoNLL2002
│       ├── dev
│       ├── ner_dataset.csv
│       ├── test
│       ├── train
│       └── vocab
├── experiments                 - this folder contains each experiment's data. 
├── models                      - this folder contains different models implemented in a single .py file.
│   ├── __init__.py
│   ├── baseline_model.py       - the very base model
│   └── model_utils             - this module contains some utility functions to help building the model.
│       └── __init__.py
│       └── rnn_layer.py
├── conll_preprocess.py         - this file contains some functions to be used in process specific raw data.
├── input_fn.py                 - some functions to load processed data and build batched input using tf.data APIs.
├── run.py                      - here is the main entrance of this project.
├── utils.py                    - some utility functions.
└── vocab.py                    - here's the Vocab class .
```

## Future work
- Finish the todo list
- Implements more models in some highlight papers.
- Abstract the project template, and build another repository for me to reuse the template easily.






