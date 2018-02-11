import urllib.request
import os
import sys
import zipfile
import numpy as np

GLOVE_EMBEDDING_SIZE = 100


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove(data_dir_path=None):
    if data_dir_path is None:
        data_dir_path = 'very_large_data'
    glove_model_path = data_dir_path + "/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
    if not os.path.exists(glove_model_path):

        glove_zip = data_dir_path + '/glove.6B.zip'

        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall(data_dir_path)
        zip_ref.close()


def load_glove(data_dir_path=None):
    if data_dir_path is None:
        data_dir_path = 'very_large_data'
    download_glove(data_dir_path)
    _word2em = {}
    glove_model_path = data_dir_path + "/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
    file = open(glove_model_path, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


def glove_zero_emb():
    return np.zeros(shape=GLOVE_EMBEDDING_SIZE)


class Glove(object):

    word2em = None

    GLOVE_EMBEDDING_SIZE = GLOVE_EMBEDDING_SIZE

    def __init__(self):
        self.word2em = load_glove()
