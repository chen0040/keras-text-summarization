from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.rnn import RecursiveRNN2
from keras_text_summarization.library.applications.fake_news_loader import fit_text
import numpy as np

LOAD_EXISTING_WEIGHTS = False


def main():
    np.random.seed(42)
    data_dir_path = './data'
    report_dir_path = './reports'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # df = df.loc[df.index < 1000]

    print('extract configuration from input texts ...')
    Y = df.title
    X = df['text']
    config = fit_text(X, Y)

    print('configuration extracted from input texts ...')

    summarizer = RecursiveRNN2(config)

    if LOAD_EXISTING_WEIGHTS:
        weight_file_path = RecursiveRNN2.get_weight_file_path(model_dir_path=model_dir_path)
        summarizer.load_weights(weight_file_path=weight_file_path)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('demo size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=20, batch_size=256)

    history_plot_file_path = report_dir_path + '/' + RecursiveRNN2.model_name + '-history.png'
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = report_dir_path + '/' + RecursiveRNN2.model_name + '-history-v' + str(summarizer.version) + '.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()