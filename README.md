# keras-text-summarization

Text summarization using seq2seq and encoder-decoder recurrent networks in Keras

# Machine Learning Models

The follow neural network models are implemented and studied for text summarization:

### Seq2Seq

The seq2seq models encodes the content of an article (encoder input) and one character (decoder input) from the summarized text to predict the next character in the summarized text

The implementation can be found in [keras_text_summarization/library/seq2seq.py](keras_text_summarization/library/seq2seq.py)

There are three variants of seq2seq model implemented for the text summarization   
* Seq2SeqSummarizer (one hot encoding)
    * training: run [demo/seq2seq_train.py](demo/seq2seq_train.py ) 
    * prediction: demo code is available in [demo/seq2seq_predict.py](demo/seq2seq_predict.py) 
* Seq2SeqGloVeSummarizer (GloVe encoding for encoder input)
    * training: run [demo/seq2seq_glove_train.py](demo/seq2seq_glove_train.py) 
    * prediction: demo code is available in [demo/seq2seq_glove_predict.py](demo/seq2seq_glove_predict.py) 
* Seq2SeqGloVeSummarizerV2 (GloVe encoding for both encoder input and decoder input)
    * training: run [demo/seq2seq_glove_v2_train.py](demo/seq2seq_glove_v2_train.py)
    * prediction: demo code is available in [demo/seq2seq_glove_v2_predict.py](demo/seq2seq_glove_v2_predict.py) 
    
### Other RNN models

There are currently 3 other encoder-decoder recurrent models based on some recommendation [here](https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/)

The implementation can be found in [keras_text_summarization/library/rnn.py](keras_text_summarization/library/rnn.py)

* One-Shot RNN (OneShotRNN in [rnn.py](keras_text_summarization/library/rnn.py)):
The one-shot RNN is a very simple encoder-decoder recurrent network model which encodes the content of an article and decodes the entire content of the summarized text
    * training: run [demo/one_hot_rnn_train.py](demo/one_hot_rnn_train.py)
    * prediction: run [demo/one_hot_rnn_predict.py](demo/one_hot_rnn_predict.py)
* Recursive RNN 1 (RecursiveRNN1 in [rnn.py](keras_text_summarization/library/rnn.py)):
The recursive RNN 1 takes the artcile content and the current built-up summarized text to predict the next character of the summarized text.
    * training: run [demo/recursive_rnn_v1_train.py](demo/recursive_rnn_v1_train.py)
    * prediction: run [demo/recursive_rnn_v1_predict.py](demo/recursive_rnn_v1_predict.py)
* Recursive RNN 2 (RecursiveRNN2 in [rnn.py](keras_text_summarization/library/rnn.py)):
The recursive RNN 2 takes the article content and the current built-up summarized text to predict the next character of the summarized text + one layer of LSTM decoder.
    * training: run [demo/recursive_rnn_v2_train.py](demo/recursive_rnn_v2_train.py)
    * prediction: run [demo/recursive_rnn_v2_predict.py](demo/recursive_rnn_v2_predict.py)

The trained models are available in the demo/models folder 

# Usage

The demo below shows how to use seq2seq to do training and prediction, but other models described above also follow
the same process of training and prediction.

### Train Deep Learning model

To train a deep learning model, say Seq2SeqSummarizer, run the following commands:

```bash
pip install requirements.txt

cd demo
python seq2seq_train.py 
```

The training code in seq2seq_train.py is quite straightforward and illustrated below:

```python
from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
from keras_text_summarization.library.applications.fake_news_loader import fit_text
import numpy as np

LOAD_EXISTING_WEIGHTS = True

np.random.seed(42)
data_dir_path = './data'
report_dir_path = './reports'
model_dir_path = './models'

print('loading csv file ...')
df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

print('extract configuration from input texts ...')
Y = df.title
X = df['text']

config = fit_text(X, Y)

summarizer = Seq2SeqSummarizer(config)

if LOAD_EXISTING_WEIGHTS:
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=100)

history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history.png'
if LOAD_EXISTING_WEIGHTS:
    history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history-v' + str(summarizer.version) + '.png'
plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})
```

After the training is completed, the trained models will be saved as cf-v1-*.* in the video_classifier/demo/models.

### Summarization

To use the trained deep learning model to summarize an article, the following code demo how to do this:

```python

from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import numpy as np

np.random.seed(42)
data_dir_path = './data' # refers to the demo/data folder
model_dir_path = './models' # refers to the demo/models folder

print('loading csv file ...')
df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
X = df['text']
Y = df.title

config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()

summarizer = Seq2SeqSummarizer(config)
summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

print('start predicting ...')
for i in range(20):
    x = X[i]
    actual_headline = Y[i]
    headline = summarizer.summarize(x)
    print('Article: ', x)
    print('Generated Headline: ', headline)
    print('Original Headline: ', actual_headline)
```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 



