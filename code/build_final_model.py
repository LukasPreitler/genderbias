import random
import statistics
import math
import numpy as np
import pandas as pd
from colorama import Fore
from colorama import Style
from time import time
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from more_itertools import random_permutation, distribute
from gensim.models.callbacks import CallbackAny2Vec
import pickle
import multiprocessing
from gensim.models import Word2Vec

sns.set_style("darkgrid")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{Fore.GREEN}------------------------------- Start, {name} --------------------------------------------------{Style.RESET_ALL}')  # Press Strg+F8 to toggle the breakpoint.

def print_step(step):
    print(f'{Fore.GREEN}------------------------------- Next step is {step} ---------------------------------------------{Style.RESET_ALL}')

def print_inter_step(inter_step):
    print(f'{Fore.YELLOW}------------------------------- {inter_step} ---------------------------------------------{Style.RESET_ALL}')

def print_done():
    print(f'{Fore.GREEN}------------------------------- Script is finished ---------------------------------------------{Style.RESET_ALL}')

def print_progress(row):
    if row % 100000 == 0 and row != 0:
        print(f'{Fore.YELLOW}#{Style.RESET_ALL}')
    elif row % 1000 == 0:
        print(f'{Fore.YELLOW}#{Style.RESET_ALL}', end="")

def print_cosim(model, list1, list2):
    d = []
    for wrd in list2:
        cos_list = []
        for i in list1:
            try:
                cos = model.wv.similarity(wrd, i)
                cos_list.append(cos)
            except:
                print(f"{i} is not in the corpura")

        d.append(cos_list)
    df = pd.DataFrame(data=np.array(d), index=list1, columns=list2)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #print(df)
    return df

def build_s(model, listW, listM, listF):
    listResult = []
    for x in listW:
        w_m = []
        w_f = []
        for m in listM:
            w_m.append(model.wv.similarity(x, m))
        for f in listF:
            w_f.append(model.wv.similarity(x, f))
        listResult.append(statistics.mean(w_m) - statistics.mean(w_f))
    return listResult

def mean_similarity(model, wlist):
    store = []
    for wrd in wlist:
        for wrd2 in wlist:
            if wrd != wrd2:
                store.append(model.wv.similarity(wrd, wrd2))
    return statistics.mean(store)

def randomWordsList(model, anzahl):
    controll = []
    a_s = 1
    b_s = 1
    both = 1
    print(f"look for independent words")
    breakcount = 0
    while breakcount < 15 and (abs(a_s) > 0.174 or abs(b_s) > 0.174 or abs(both) > 0.174):

        controll.clear()
        dict_key = schoolbookmodel.wv.key_to_index.keys()
        len_con = anzahl
        for i in range(len_con):
            key = random.choice(list(dict_key))
            controll.append(key)
        alist, blist = distribute(2, controll)

        a_s = mean_similarity(model, alist)
        b_s = mean_similarity(model, blist)
        both = mean_similarity(model, controll)
        breakcount = breakcount + 1
    if breakcount > 10:
        print(f"{Fore.YELLOW}Found: alist sim = {a_s}, blist sim = {b_s}, sim over all = {both}{Fore.WHITE}")
    else:
        print(f"Found: alist sim = {a_s}, blist sim = {b_s}, sim over all = {both}")


    return controll, breakcount, a_s, b_s, both

def cohesD(listM, listF, controll):
    s_m = build_s(schoolbookmodel, listM, controll[:len(listM)], controll[len(listF):])
    s_f = build_s(schoolbookmodel, listF, controll[:len(listM)], controll[len(listF):])

    cohens_top = (statistics.mean(s_m) - statistics.mean(s_f))

    s_w = build_s(schoolbookmodel, listM + listF, controll[:len(listM)], controll[len(listF):])

    s_w_mean = statistics.mean(s_w)
    s_botom = 0
    for elem in s_w:
        s_botom = s_botom + pow(elem - s_w_mean, 2)
    cohens_bottom = math.sqrt(s_botom/(len(s_w)))
    cohensd = cohens_top/cohens_bottom
    print(f"cohen's D {cohensd}")
    return cohensd


def permutationtest(model, threadment, controll, iter):
    print(f"random select controll words are: {controll}")

    uList = threadment + controll
    print(f"lenght of ulist: {len(uList)} {uList}")

    test_permutation = []
    test_threadment = []

    for rwrd in threadment:
        for cwrd in threadment:
            if rwrd == cwrd:
                continue
            else:
                test_threadment.append(model.wv.similarity(rwrd, cwrd))

    test_value = statistics.mean(test_threadment)

    for iteration in range(0, iter):
        print_progress(iteration)
        plist = (random_permutation(uList, len(uList)))
        x, y = distribute(2, plist)
        x = list(x)
        y = list(y)
        x_store = []
        y_store = []
        for rwrd in x:
            for cwrd in x:
                if rwrd == cwrd:
                    continue
                else:
                    x_store.append(model.wv.similarity(rwrd, cwrd))
        for rwrd in y:
            for cwrd in y:
                if rwrd == cwrd:
                    continue
                else:
                    y_store.append(model.wv.similarity(rwrd, cwrd))
        test_permutation.append(statistics.mean(x_store))
        test_permutation.append(statistics.mean(y_store))

    count_greater = 0
    for test in test_permutation:
        if test >= test_value:
            count_greater = count_greater + 1

    print(f"\nTest statistik: {test_value} observed \n"
          f"p-value: {count_greater/len(test_permutation)} = {count_greater} / {len(test_permutation)}")
    return count_greater/len(test_permutation)  # p-value


def tsnescatterplot(model, target, femalelist, malelist):
    arrays = np.empty((0, 500), dtype="f")
    word_labels = [target]
    color_list = ["red"]

    arrays = np.append(arrays, model.wv.__getitem__([target]), axis=0)

    for wrd in femalelist:
        word_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('blue')
        arrays = np.append(arrays, word_vector, axis=0)

    for wrd in malelist:
        word_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, word_vector, axis=0)

    reduc = PCA(n_components=13).fit_transform(arrays)

    np.set_printoptions(suppress=True)

    x_tsne = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    df = pd.DataFrame({"x": [x for x in x_tsne[:, 0]],
                       "y": [y for y in x_tsne[:, 1]],
                       "words": word_labels,
                       "color": color_list})

    fig_sc, _ = plt.subplots()
    fig_sc.set_size_inches(9, 9)

    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolor': df['color']
                                  }
                     )

    for line in range(0, df.shape[0]):
        p1.text(df['x'][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

        plt.xlim(x_tsne[:, 0].min() - 50, x_tsne[:, 0].max() + 50)
        plt.ylim(x_tsne[:, 1].min() - 50, x_tsne[:, 1].max() + 50)

        plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.show()

def checkGenderPairs(flist, mlist, oflist, omlist):
    print(flist)
    print(mlist)
    fem = []
    mal = []
    for wrd in oflist:
        if wrd in flist:
            fem.append(1)
        else:
            fem.append(0)

    for wrd in omlist:
        if wrd in mlist:
            mal.append(1)
        else:
            mal.append(0)
    mal_x = []
    fem_x = []
    for i in range(len(fem)):
        if fem[i] == 1 and mal[i] == 1:
            mal_x.append(omlist[i])
            fem_x.append(oflist[i])

    print(f"new male list: {mal_x}")
    print(f"new female list: {fem_x}")
    return fem_x, mal_x

class EpochLogger(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch), end=" ")
        preLosses = model.get_latest_training_loss()
        model.running_training_loss = 0
        print(f"preLosses {preLosses}", end=" -> ")

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch), end=" ")
        postLosses = model.get_latest_training_loss()
        print(f"postlosses = {postLosses}")

        self.epoch += 1

if __name__ == '__main__':
    print_hi('creates model')

    folder_save = "model/random/final/"
    os.makedirs(f'{folder_save}', exist_ok=True)
    start_time = time()

    print_step("Build model")
    # source: https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook#Gensim-Word2Vec-Implementation:

    cores = multiprocessing.cpu_count()

    print_step("load text")

    with open(f'corpora/cleaned_text.pickle', 'rb') as f:
        sentences_token_cleaned = pickle.load(f)

    data = []
    progress = []
    failed = []

    # need to be set for your model
    interations = 90
    minWord = 5
    windowSize = 1
    vs = 300
    neg = 1

    print('Time since start'.format(round((time() - start_time) / 60, 2)))
    print_step(f"Neue Model: min={minWord}, window={windowSize}, vector={vs}, neg={neg}")
    schoolbookmodel = Word2Vec(min_count=minWord,
                               window=windowSize,
                               vector_size=vs,
                               negative=neg,
                               workers=cores - 1)

    corpura_count = 0
    for sent in sentences_token_cleaned:
        for word in sent:
            corpura_count = corpura_count + 1

    print_inter_step(f"build vocab from {corpura_count} words")

    t = time()

    schoolbookmodel.build_vocab(sentences_token_cleaned, progress_per=1000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    epoch_logger = EpochLogger()
    schoolbookmodel.train(sentences_token_cleaned,
                            total_examples=schoolbookmodel.corpus_count,
                            epochs=interations,
                            report_delay=1,
                            compute_loss=True,
                            callbacks=[epoch_logger])

    print_inter_step(f"validate")

    losses = schoolbookmodel.get_latest_training_loss()
    print(f"Delta Losses: {losses}")

    print(f"Model: {schoolbookmodel}")
    print_step(f'model is created, now save it')
    schoolbookmodel.save(f"{folder_save}model")

    print_done()
