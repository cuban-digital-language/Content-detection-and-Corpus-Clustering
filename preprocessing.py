import os
import ast
from matplotlib import pyplot as plt
import progressbar
from sklearn.decomposition import TruncatedSVD
from tokenizer.custom_tokenizer import CustomToken, SpacyCustomTokenizer
import sys
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer


def files_list(folder):
    return list(os.listdir(f'{folder}'))


def load(files, path=''):
    for filename in files:
        with open(f'{path}/{filename}', 'r') as f:
            text = f.read()
            f.close()
            yield ast.literal_eval(text)


def pb(len_, name=""):
    bar = progressbar.ProgressBar(len_, widgets=[progressbar.Bar(
        '=', '[', ']', ), name, progressbar.Percentage()])
    bar.start()
    return bar


def data_to_token(data):
    token = CustomToken(data["text"], lex=data['lemma'],
                        is_stop=data['is_stop'], is_sy=data['is_symbol'])
    return token


def doc2vec(files):
    document = []
    matrix = []
    vocabulary = set()
    nlp = SpacyCustomTokenizer()
    _len_ = len(files)
    bar = pb(_len_, f' tokenizer {_len_} ')
    for i, data in enumerate(load(files, 'tokens')):
        matrix.append(nlp.nlp(data[0]).vector)
        s = set()
        for token in data[1]:
            token = data_to_token(token)

            if (token.is_stop
                or token.is_symbol
                or token.space()
                or token.is_emoji()
                or token.is_url()
                # or token.is_date()
                    or token.is_digit()):
                continue
            lemma = token.lemma.lower()
            vocabulary.add(lemma)
            s.add(lemma)
            # s += lemma + " "
        document.append(s)
        bar.update(i+1)
    bar.finish()

    # tf = TfidfVectorizer()
    # matrix = tf.fit_transform(document)

    # if matrix.shape > (len(document), 96):
    #     print("SVD decomposition")
    #     truncatedSVD = TruncatedSVD(96)
    #     matrix = truncatedSVD.fit_transform(matrix)

    return matrix, document


def save(vectors, document):

    np.save('results/vectors.npy', vectors)

    with open('results/document.json', 'w+') as f:
        json.dump(document, f)
        f.close()


def loads():

    vectors = np.load('results/vectors.npy', allow_pickle=True)
    l = len(vectors[0])
    vectors = [i for i in vectors if len(i) == l]

    with open('results/document.json', 'r') as f:
        document = json.load(f)
        f.close()

    return vectors, document


def view_points(vectors, tags=None):
    truncatedSVD = TruncatedSVD(2)
    X_truncate_plot = truncatedSVD.fit_transform(vectors)

    x = [point[0] for point in X_truncate_plot]
    y = [point[1] for point in X_truncate_plot]
    plt.scatter(x, y, c=tags)
    plt.show()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) == 2 else ''
    if cmd == 'plot':
        print("ONLY PLOT")
        v, _ = loads()
    else:
        files = files_list('tokens')
        v, d = doc2vec(files)
        save(v, d)
    v, _ = loads()
    view_points(v)
