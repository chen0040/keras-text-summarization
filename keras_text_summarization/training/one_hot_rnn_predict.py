from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.rnn import OneShotRNN
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
    # df = df.loc[df.index < 1000]
    X = df['text']
    Y = df.title

    config = np.load(OneShotRNN.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = OneShotRNN(config)
    summarizer.load_weights(weight_file_path=OneShotRNN.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for i in np.random.permutation(np.arange(len(X)))[0:20]:
        x = X[i]
        actual_headline = Y[i]
        headline = summarizer.summarize(x)
        # print('Article: ', x)
        print('Generated Headline: ', headline)
        print('Original Headline: ', actual_headline)


if __name__ == '__main__':
    main()
