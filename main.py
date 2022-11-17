from work_with_text import create_list_all_sentences

import pandas as pd
import warnings
import numpy as np
from gensim import models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

warnings.filterwarnings("ignore")


def tsne_scatterplot(model, word, info):
    vectors_words = [model.wv.word_vec(word)]
    word_labels = [word]
    color_list = ['red']
    close_words = model.wv.most_similar(word)
    for wrd_score in close_words:
        wrd_vector = model.wv.word_vec(wrd_score[0])
        vectors_words.append(wrd_vector)
        word_labels.append(wrd_score[0])
        color_list.append('blue')
    # t-SNE reduction
    vectors_words = np.asarray(vectors_words)
    Y = (TSNE(n_components=2, random_state=0, perplexity=10, init="pca").fit_transform(vectors_words))
    # Sets everything up to plot
    df = pd.DataFrame({"x": [x for x in Y[:, 0]], "y": [y for y in Y[:, 1]], "words": word_labels, "color": color_list})
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    p1 = sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", scatter_kws={"s": 40, "facecolors": df["color"]})

    # Adds annotations one by one with a loop

    for line in range(0, df.shape[0]):
        p1.text(df["x"][line], df["y"][line], " " + df["words"][line].title(), horizontalalignment="left",
                verticalalignment="bottom", size="medium", color=df["color"][line], weight="normal").set_size(15)
    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)
    plt.title('t-SNE visualization for {}'.format(word.title()) + info)
    plt.show()


with open("ap/ap.txt", 'r', encoding='utf8') as text:
    main_text = text.read()
all_sentences = create_list_all_sentences(main_text)
word2vec = models.word2vec.Word2Vec(all_sentences, min_count=2, vector_size=20, sg=0)
word2vec_2 = models.word2vec.Word2Vec(all_sentences, min_count=3, vector_size=15, sg=0)
word2vec_3 = models.word2vec.Word2Vec(all_sentences, min_count=4, vector_size=100, sg=0)

word2vec_words = word2vec.wv.most_similar('word')
word2vec_2_words = word2vec_2.wv.most_similar('word')
word2vec_3_words = word2vec_3.wv.most_similar('word')

tsne_scatterplot(word2vec, word2vec_words[5][0], info = 'min_count = 2, vector_size = 20')
tsne_scatterplot(word2vec_2, word2vec_2_words[5][0], info = 'min_count = 3, vector_size = 10')
tsne_scatterplot(word2vec_3, word2vec_3_words[5][0], info = 'min_count = 4, vector_size = 100')

