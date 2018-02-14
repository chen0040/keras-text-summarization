from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.seq2seq import Seq2SeqGloVeSummarizerV2
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'
    very_large_data_dir_path = './very_large_data'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
    X = df['text']
    Y = df.title

    config = np.load(Seq2SeqGloVeSummarizerV2.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = Seq2SeqGloVeSummarizerV2(config)
    summarizer.load_glove(very_large_data_dir_path)
    summarizer.load_weights(weight_file_path=Seq2SeqGloVeSummarizerV2.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for i in np.random.permutation(np.arange(len(X)))[0:20]:
        x = X[i]
        actual_headline = Y[i]
        headline = summarizer.summarize(x)

        print('Generated Headline: ', headline)
        print('Original Headline: ', actual_headline)
        # print('Article: ', x[0:100])


if __name__ == '__main__':
    main()
