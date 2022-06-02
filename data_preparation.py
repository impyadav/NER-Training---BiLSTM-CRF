"""
# -*- coding: utf-8 -*-

Created on Jun 2022
@author: Prateek Yadav

"""
import pandas as pd


class nerDataPrep:

    def __init__(self, csv_file_loc):
        self.file = csv_file_loc

    def get_df(self):
        return pd.read_csv(self.file)

    def print_info(self):
        df = self.get_df()
        print('Dataset len: ', len(df))
        word_counts = df.groupby('Sentence #')['Word'].agg(['count'])
        MAX_LEN = max(word_counts['count'].values)
        print('MAX_LEN: ', MAX_LEN)

    def create_dict(self):
        df = self.get_df()
        print('Dataset len: ', len(df))
        word_counts = df.groupby('Sentence #')['Word'].agg(['count'])
        all_words = list(set(df.Word.values))
        all_tags = list(set(df.Tag.values))
        max_len = max(word_counts['count'].values)

        # word Dict building
        word2index = {word: idx + 2 for idx, word in enumerate(all_words)}
        word2index['UNK'] = 0
        word2index['PAD'] = 1
        index2word = {idx: word for word, idx in word2index.items()}

        # tag dict Building
        tag2index = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
        tag2index['PAD'] = 0
        index2tag = {idx: tag for tag, idx in tag2index.items()}

        return df, word2index, index2word, tag2index, index2tag, max_len

    def create_tuple(self, dataframe):
        iterator = zip(dataframe.Word.values.tolist(),
                       dataframe.POS.values.tolist(),
                       dataframe.Tag.values.tolist())
        return [(word, pos, tag) for word, pos, tag in iterator]


if __name__ == '__main__':
    dataPrepObj = nerDataPrep('../data/something.csv')
    dataPrepObj.print_info()
    df, word2idx, idx2word, tag2idx, idx2tag, MAX_LEN = dataPrepObj.create_dict()
    sentences = df.groupby('Sentence #').apply(dataPrepObj.create_tuples).tolist()

    X = [[item[0] for item in sentence] for sentence in sentences]
    y = [[item[2] for item in sentence] for sentence in sentences]

    X = [[word2idx[word] for word in sentence] for sentence in X]
    y = [[tag2idx[tag] for tag in sentence] for sentence in y]

    print('X shape:', X.shape)
    print('y shape: ', y.shape)

