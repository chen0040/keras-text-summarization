from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.seq2seq import Seq2Seq
from keras_text_summarization.utility.fake_news_loader import fit_text
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = './models'

    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # Set `y`
    Y = df.title

    # Drop the `label` column
    df.drop("title", axis=1)

    print('extract configuration from input texts ...')

    X = df['text']

    config = fit_text(X, Y)

    print('configuration extracted from input texts ...')

    classifier = Seq2Seq(config)
    classifier.load_weights(weight_file_path=Seq2Seq.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for x in X[0:20]:
        headline = classifier.summarize(x)
        print(headline)



if __name__ == '__main__':
    main()
