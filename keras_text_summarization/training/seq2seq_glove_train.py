from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.seq2seq import Seq2SeqGloVe
from keras_text_summarization.utility.fake_news_loader import fit_text
import numpy as np

LOAD_EXISTING_WEIGHTS = False


def main():
    np.random.seed(42)
    data_dir_path = './data'
    very_large_data_dir_path ='./very_large_data'
    report_dir_path = './reports'
    model_dir_path = './models'

    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # Set `y`
    Y = df.title

    # Drop the `title` column
    df.drop("title", axis=1)

    print('extract configuration from input texts ...')

    X = df['text']

    config = fit_text(X, Y)

    print('configuration extracted from input texts ...')

    classifier = Seq2SeqGloVe(config)
    classifier.load_glove(very_large_data_dir_path)

    if LOAD_EXISTING_WEIGHTS:
        classifier.load_weights(weight_file_path=Seq2SeqGloVe.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = classifier.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=100)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqGloVe.model_name + '-history.png'
    plot_and_save_history(history, classifier.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()