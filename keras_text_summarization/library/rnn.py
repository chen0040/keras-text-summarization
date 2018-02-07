from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Embedding, Dense, Input, RepeatVector, TimeDistributed, concatenate, Merge, add
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

HIDDEN_UNITS = 100
BATCH_SIZE = 64
VERBOSE = 1
EPOCHS = 10


class OneShotRNN(object):
    model_name = 'one-shot-rnn'
    """
    The first alternative model is to generate the entire output sequence in a one-shot manner.
    That is, the decoder uses the context vector alone to generate the output sequence.
    
    This model puts a heavy burden on the decoder.
    It is likely that the decoder will not have sufficient context for generating a coherent output sequence as it 
    must choose the words and their order.
    """

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config
        self.version = 0
        if 'version' in config:
            self.version = config['version']

        print('max_input_seq_length', self.max_input_seq_length)
        print('max_target_seq_length', self.max_target_seq_length)
        print('num_input_tokens', self.num_input_tokens)
        print('num_target_tokens', self.num_target_tokens)

        # encoder input model
        model = Sequential()
        model.add(Embedding(output_dim=128, input_dim=self.num_input_tokens, input_length=self.max_input_seq_length))

        # encoder model
        model.add(LSTM(128))
        model.add(RepeatVector(self.max_target_seq_length))
        # decoder model
        model.add(LSTM(128, return_sequences=True))
        model.add(TimeDistributed(Dense(self.num_target_tokens, activation='softmax')))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_target_data_batch[lineIdx, idx, w2idx] = 1
                yield encoder_input_data_batch, decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + OneShotRNN.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + OneShotRNN.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + OneShotRNN.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None, batch_size=None):
        if epochs is None:
            epochs = EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version

        config_file_path = OneShotRNN.get_config_file_path(model_dir_path)
        weight_file_path = OneShotRNN.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = OneShotRNN.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        predicted = self.model.predict(input_seq)
        predicted_word_idx_list = np.argmax(predicted, axis=1)
        predicted_word_list = [self.target_idx2word[wid] for wid in predicted_word_idx_list[0]]
        return predicted_word_list


class RecursiveRNN1(object):
    model_name = 'recursive-rnn-1'
    """
    A second alternative model is to develop a model that generates a single word forecast and call it recursively.
    
    That is, the decoder uses the context vector and the distributed representation of all words generated so far as 
    input in order to generate the next word. 
    
    A language model can be used to interpret the sequence of words generated so far to provide a second context vector 
    to combine with the representation of the source document in order to generate the next word in the sequence.
    
    The summary is built up by recursively calling the model with the previously generated word appended (or, more 
    specifically, the expected previous word during training).
    
    The context vectors could be concentrated or added together to provide a broader context for the decoder to 
    interpret and output the next word.
    """

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        if 'version' in config:
            self.version = config['version']
        else:
            self.version = 0
        self.config = config

        print('max_input_seq_length', self.max_input_seq_length)
        print('max_target_seq_length', self.max_target_seq_length)
        print('num_input_tokens', self.num_input_tokens)
        print('num_target_tokens', self.num_target_tokens)

        inputs1 = Input(shape=(self.max_input_seq_length,))
        am1 = Embedding(self.num_input_tokens, 128)(inputs1)
        am2 = LSTM(128)(am1)

        inputs2 = Input(shape=(self.max_target_seq_length,))
        sm1 = Embedding(self.num_target_tokens, 128)(inputs2)
        sm2 = LSTM(128)(sm1)

        decoder1 = concatenate([am2, sm2])
        outputs = Dense(self.num_target_tokens, activation='softmax')(decoder1)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def split_target_text(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        encoder_input_data_batch = []
        decoder_input_data_batch = []
        decoder_target_data_batch = []
        line_idx = 0
        while True:
            for recordIdx in range(0, len(x_samples)):
                y = y_samples[recordIdx]
                x = x_samples[recordIdx]
                for target_words in enumerate(y):
                    decoder_input_line = []

                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        decoder_input_line.append(w2idx)
                        decoder_target_label = np.zeros(self.num_target_tokens)
                        if w2idx != 0:
                            if idx != 0:
                                decoder_target_label[w2idx] = 1
                        decoder_input_data_batch.append(np.array(decoder_input_line))
                        encoder_input_data_batch.append(x)
                        decoder_target_data_batch.append(decoder_target_label)

                line_idx += 1
                if line_idx >= batch_size:
                    yield [pad_sequences(encoder_input_data_batch, self.max_input_seq_length),
                           pad_sequences(decoder_input_data_batch,
                                         self.max_target_seq_length)], np.array(decoder_target_data_batch)
                    line_idx = 0
                    encoder_input_data_batch = []
                    decoder_input_data_batch = []
                    decoder_target_data_batch = []

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN1.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN1.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN1.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None, batch_size=None):
        if epochs is None:
            epochs = EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version

        config_file_path = RecursiveRNN1.get_config_file_path(model_dir_path)
        weight_file_path = RecursiveRNN1.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = RecursiveRNN1.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.split_target_text(Ytrain)
        Ytest = self.split_target_text(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        sum_input_seq = np.zeros(
            shape=(1, self.max_target_seq_length))
        sum_input_seq[0, 0] = self.target_word2idx['START']
        terminated = False
        target_text_len = 0
        target_text = ''
        while not terminated:
            output_tokens = self.model.predict([input_seq, sum_input_seq])

            sample_token_idx = np.argmax(output_tokens[0, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True
            else:
                sum_input_seq[0, target_text_len] = sample_token_idx
        return target_text


class RecursiveRNN2(object):
    model_name = 'recursive-rnn-2'
    """
    In this third alternative, the Encoder generates a context vector representation of the source document.

    This document is fed to the decoder at each step of the generated output sequence. This allows the decoder to build 
    up the same internal state as was used to generate the words in the output sequence so that it is primed to generate 
    the next word in the sequence.

    This process is then repeated by calling the model again and again for each word in the output sequence until a 
    maximum length or end-of-sequence token is generated.
    """

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        self.version = 0
        if 'version' in config:
            self.version = config['version']

        # article input model
        inputs1 = Input(shape=(self.max_input_seq_length,))
        article1 = Embedding(self.num_input_tokens, 128)(inputs1)
        article2 = LSTM(128)(article1)
        article3 = RepeatVector(self.max_target_seq_length)(article2)
        # summary input model
        inputs2 = Input(shape=(self.max_target_seq_length,))
        summ1 = Embedding(self.num_target_tokens, 128)(inputs2)
        # decoder model
        decoder1 = concatenate([article3, summ1])
        decoder2 = LSTM(128)(decoder1)
        outputs = Dense(self.num_target_tokens, activation='softmax')(decoder2)
        # tie it together [article, summary] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def split_target_text(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples):
        encoder_input_data_batch = []
        decoder_input_data_batch = []
        decoder_target_data_batch = np.zeros(
            shape=(BATCH_SIZE, self.max_target_seq_length, self.num_target_tokens))
        line_idx = 0
        while True:
            for recordIdx in range(0, len(x_samples)):
                y = y_samples[recordIdx]
                x = x_samples[recordIdx]
                for target_words in enumerate(y):
                    decoder_line = []

                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        decoder_line.append(w2idx)
                        if w2idx != 0:
                            if idx != 0:
                                decoder_target_data_batch[line_idx, idx-1, w2idx] = 1
                        decoder_input_data_batch.append(decoder_line)
                        encoder_input_data_batch.append(x)
                        line_idx += 1
                        if line_idx >= BATCH_SIZE:
                            yield [pad_sequences(encoder_input_data_batch, self.max_input_seq_length),
                                   pad_sequences(decoder_input_data_batch, self.max_target_seq_length)], decoder_target_data_batch
                            line_idx = 0
                            encoder_input_data_batch = []
                            decoder_input_data_batch = []
                            decoder_target_data_batch = np.zeros(
                                shape=(BATCH_SIZE, self.max_target_seq_length, self.num_target_tokens))


    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN2.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN2.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + RecursiveRNN2.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None):
        if epochs is None:
            epochs = EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'

        self.version += 1
        self.config['version'] = self.version

        config_file_path = RecursiveRNN2.get_config_file_path(model_dir_path)
        weight_file_path = RecursiveRNN2.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = RecursiveRNN2.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.split_target_text(Ytrain)
        Ytest = self.split_target_text(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain)
        test_gen = self.generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain) * self.max_target_seq_length // BATCH_SIZE
        test_num_batches = len(Xtest) * self.max_target_seq_length // BATCH_SIZE

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        sum_input_seq = np.zeros(
            shape=(1, self.max_target_seq_length))
        sum_input_seq[0, 0] = self.target_word2idx['START']
        terminated = False
        target_text_len = 0
        target_text = ''
        while not terminated:
            output_tokens = self.model.predict([input_seq, sum_input_seq])

            sample_token_idx = np.argmax(output_tokens[0, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True
            else:
                sum_input_seq[0, target_text_len] = sample_token_idx
        return target_text
