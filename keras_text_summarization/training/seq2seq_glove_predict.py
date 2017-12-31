from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.seq2seq import Seq2SeqGloVe
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'
    very_large_data_dir_path = './very_large_data'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
    X = df['text']

    config = np.load(Seq2SeqGloVe.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = Seq2SeqGloVe(config)
    summarizer.load_glove(very_large_data_dir_path)
    summarizer.load_weights(weight_file_path=Seq2SeqGloVe.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for x in X[0:20]:
        headline = summarizer.summarize(x)
        print(headline)


if __name__ == '__main__':
    main()
