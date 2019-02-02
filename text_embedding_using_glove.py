from __future__ import print_function

import numpy as np
import codecs


def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        for line in f:
            splitlines = line.split()
            vocab_write.write(splitlines[0].strip())
            vocab_write.write("\n")
            wv.append([float(val) for val in splitlines[1:]])
    np.save(embedding_path + ".npy", np.array(wv))


def load_glove_model(glove_file):
    """

    :param gloveFile: embeddings_path: path of glove file.
    :return: glove model
    """

    f = codecs.open(glove_file + ".txt", 'r', encoding='utf-8')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    return model


def load_embeddings_binary(embeddings_path):
    """
    It loads embedding provided by glove which is saved as binary file. Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embeddings_path + '.npy')
    model = {}
    for i, w in enumerate(index2word):
        model[w] = wv[i]
    return model


def get_w2v(sentence, model):
    """

    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words in input sentence.
    """
    return np.array([model.get(val, np.zeros(100)) for val in sentence.split()], dtype=np.float64)


def read_get_dataset():
    """

    :return: reads dataset, removes punctuation and return.
    """
    import string

    f = open("input.txt")
    s = f.read()
    s.translate(None, string.punctuation)

    return s


def main():
    pass

if __name__ == '__main__':
    import datetime
    a = datetime.datetime.now()
    # Download glove vector from stanford link provided in readme
    glove_model_1 = load_glove_model("glove/glove.6B.100d")
    b = datetime.datetime.now()
    print("It took " + str(b - a) + " seconds to load glove model from .txt file")
    glove_model_2 = load_embeddings_binary("glove/glove.6B.100d")
    print("It took " + str(datetime.datetime.now() - b) + " seconds to load glove model from binary file")

    word_embedding = get_w2v("my name is sharmila", glove_model_2)

    print(word_embedding)
