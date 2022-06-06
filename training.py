"""
# -*- coding: utf-8 -*-

Created on Jun 2022
@author: Prateek Yadav

"""
from model import nerModel
from data_preparation import nerDataPrep

import yaml
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    dataPrepObj = nerDataPrep('../data/something.csv')
    dataPrepObj.print_info()
    df, word2idx, idx2word, tag2idx, idx2tag, MAX_LEN = dataPrepObj.create_dict()
    sentences = df.groupby('Sentence #').apply(dataPrepObj.create_tuples).tolist()

    X = [[item[0] for item in sentence] for sentence in sentences]
    y = [[item[2] for item in sentence] for sentence in sentences]

    X = [[word2idx[word] for word in sentence] for sentence in X]
    y = [[tag2idx[tag] for tag in sentence] for sentence in y]

    # update vocab stats in config

    vocab_data = {'max_len': MAX_LEN, 'tag_count': len(tag2idx), 'word_count': len(idx2word)}
    with open('config/train_config.yaml', 'w') as f:
        yaml.dump(vocab_data, f, default_flow_style=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

    # print("Number of sentences in the training dataset: {}".format(len(X_train)))
    # print("Number of sentences in the test dataset : {}".format(len(X_test)))

    X_train = np.asarray(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    # modelling

    with open('config/train_config.yaml') as f:
        configObj = yaml.load(f)

    model = nerModel(configObj)

    history = model.fit(X_train, y_train, batch_size=configObj.batch_size,
                              epochs=configObj.max_epochs, validation_split=0.1,
                              verbose=2)

    model.save(configObj.model_name)


main()
