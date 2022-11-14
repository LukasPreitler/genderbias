import random
import statistics
import math

import numpy as np
import pandas as pd
from colorama import Fore
from colorama import Style
from time import time, monotonic
import os
from multiprocessing import cpu_count

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_style("darkgrid")


from more_itertools import random_permutation, distribute

from gensim.models.callbacks import CallbackAny2Vec
from HanTa import HanoverTagger as ht

import umap

import pickle
np.set_printoptions(suppress=True)
pd.set_option("expand_frame_repr", False)


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


def prepareList(Words, model):
    doRemove = []
    for index in range(len(Words)):
        if not model.wv.has_index_for(Words[index]):
            doRemove.append(Words[index])

    for elem in doRemove:
        Words.remove(elem)

def build_s(model, listW, listM, listF):
    listResult = []
    for x in listW:
        w_m = []
        w_f = []
        for m in listM:
            if x != m:
                w_m.append(model.wv.similarity(x, m))
        for f in listF:
            if x != f:
                w_f.append(model.wv.similarity(x, f))
        listResult.append(statistics.mean(w_m) - statistics.mean(w_f))
    return listResult

def build_s(model, listW, listM, listF):
    listResult = []
    for x in listW:
        w_m = []
        w_f = []
        for m in listM:
            if x != m:
                w_m.append(model.wv.similarity(x, m))
        for f in listF:
            if x != f:
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


def mean_similarityList(model, wlist):
    store = []
    for wrd in wlist:
        for wrd2 in wlist:
            if wrd != wrd2:
                store.append(model.wv.similarity(wrd, wrd2))
    return store

def cohesD(model, listM, listF):

    mean_m = mean_similarity(model, listM)
    mean_f = mean_similarity(model, listF)
    cohens_top = (mean_m - mean_f)

    mean_f_list = mean_similarityList(model, listF)

    s_bottom = 0

    for elem in mean_f_list:
        s_bottom = s_bottom + math.pow((elem - mean_f), 2)

    cohens_bottom = math.sqrt((s_bottom/(len(mean_f_list))))

    cohensd = cohens_top/cohens_bottom
    #print(f"cohen's D {cohensd}")
    return cohensd


def setTest(model, threadment, controll, p_iterations):
    ulist = threadment + controll

    s_stat_1 = mean_similarity(model, threadment)
    s_stat_2 = mean_similarity(model, controll)
    s_stat = s_stat_1 * s_stat_2

    test_permutation = []
    for iteration in range(0, p_iterations):
        print_progress(iteration)
        plist = (random_permutation(ulist, len(ulist)))
        x = plist[len(threadment):]
        y = plist[:len(threadment)]
        s_stat_1 = mean_similarity(model, x)
        s_stat_2 = mean_similarity(model, y)
        p_test = s_stat_1 * s_stat_2
        test_permutation.append(p_test)

    count_greater = 0
    for test in test_permutation:
        if test >= s_stat:
            count_greater = count_greater + 1

    print(f"Permuation Test: t-stat={s_stat}, some tests={test_permutation[100:150]}")
    return count_greater / len(test_permutation)


def permutationtest(model, threadment, controll, iter):

    uList = threadment + controll
    #print(f"lenght of ulist: {len(uList)} {uList}")

    test_permutation = []

    mean_treatment = build_s(model, threadment, threadment, controll)
    #mean_control = build_s(model, threadment, threadment, controll)
    test_value = statistics.mean(mean_treatment)

    for iteration in range(0, iter):
        print_progress(iteration)
        plist = (random_permutation(uList, len(uList)))
        x, y = distribute(2, plist)
        frist = build_s(model, threadment, list(x), list(y))
        #second = mean_similarity_controll(model, threadment, list(y))
        test_permutation.append(statistics.mean(frist))


    count_greater = 0
    for test in test_permutation:
        if test >= test_value:
            count_greater = count_greater + 1

    print(f"\nTest value: {test_value} observed, controll values{test_permutation[100:150]} \n"
          f"p-value: {count_greater/len(test_permutation)} = {count_greater} / {len(test_permutation)}")

    #print(tstat)
    return count_greater/len(test_permutation)  # p-value


class EpochLogger(CallbackAny2Vec):

    def __init__(self, table):
        self.epoch = 0

    def on_epoch_begin(self, model):
        #print("Epoch #{} start".format(self.epoch), end=" ")
        #preLosses = model.get_latest_training_loss()
        model.running_training_loss = 0
        #print(f"preLosses {preLosses}", end=" -> ")

    def on_epoch_end(self, model):
        #print("Epoch #{} end".format(self.epoch), end=" ")
        postLosses = model.get_latest_training_loss()
        #print(f"{postLosses}", end=", ")
        table.append(postLosses)
        self.epoch += 1

if __name__ == '__main__':
    print_hi('creates model')

    print_step("load HanoverTagger for lammaziation")
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')

    print_step("Build model")
    # source: https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook#Gensim-Word2Vec-Implementation:
    import multiprocessing
    from gensim.models import Word2Vec

    cores = multiprocessing.cpu_count()

    print_step("load text")

    table = []

    folder = "model/random/top5/"
    os.makedirs(f'{folder}', exist_ok=True)
    test_epochs = []  # len() = 7
    windowSizeVec = []  # len() = 6
    vsVec = []  # len() = 7
    negVec = []  # len() = 7
    minWordVec = []

    delta_losses = pd.DataFrame(index=[0, 1, 2, 3, 4])
    total_models = 5

    frame_x_min = 0
    delta_losses = pd.read_csv(f"{folder}delta_losses5")

    delta_losses.columns = ["Epochs", "Top1", "Top2", "Top3", "Top4", "Statistically"]

    # todo: change axis and trainings iterations for your need
    lines = delta_losses["Statistically"].plot.line( xlabel="Epochs", ylabel="Delta Losses", color="black")
    fig = lines.get_figure()
    plt.title("Delta Loss of models")
    plt.xlim([frame_x_min, 100])
    plt.ylim([5000, 80000])
    plt.hlines(statistics.mean(delta_losses["Statistically"][85:90]),
               xmin=frame_x_min, xmax=100, label="Mean Statistically",
               colors="black", linestyles="--")
    delta_losses["Top1"].plot.line(color="dimgray")
    plt.hlines(statistics.mean(delta_losses["Top1"][95:100]),
               xmin=frame_x_min, xmax=100, label="Mean Top1",
               colors="dimgray", linestyles="--")
    delta_losses["Top2"].plot.line(color="gray")
    plt.hlines(statistics.mean(delta_losses["Top2"][90:95]),
               xmin=frame_x_min, xmax=100, label="Mean Top2",
               colors="gray", linestyles="--")
    delta_losses["Top3"].plot.line(color="darkgray")
    plt.hlines(statistics.mean(delta_losses["Top3"][95:100]),
               xmin=frame_x_min, xmax=100, label="Mean Top3",
               colors="darkgray", linestyles="--")
    delta_losses["Top4"].plot.line(color="lightgray")
    plt.hlines(statistics.mean(delta_losses["Top4"][85:90]),
               xmin=frame_x_min, xmax=100, label="Mean Top4",
               colors="lightgray", linestyles="--")

    plt.show()
    fig.savefig(f'{folder}deltalosses.png', bbox_inches="tight")

    # todo: change axis and trainings iterations for your need
    frame_x_min = 75
    lines = delta_losses["Statistically"].plot.line(xlabel="Epochs", ylabel="Delta Losses", color="black")

    fig_small = lines.get_figure()
    plt.xlim([frame_x_min, 100])
    plt.ylim([15000, 30000])
    plt.hlines(statistics.mean(delta_losses["Statistically"][85:90]),
               xmin=frame_x_min, xmax=100, label="Mean Statistically",
               colors="black", linestyles="--")
    delta_losses["Top1"].plot.line(color="dimgray")
    plt.hlines(statistics.mean(delta_losses["Top1"][95:100]),
               xmin=frame_x_min, xmax=100, label="Mean Top1",
               colors="dimgray", linestyles="--")
    delta_losses["Top2"].plot.line(color="gray")
    plt.hlines(statistics.mean(delta_losses["Top2"][95:100]),
               xmin=frame_x_min, xmax=100, label="Mean Top2",
               colors="gray", linestyles="--")
    delta_losses["Top3"].plot.line(color="darkgray")
    plt.hlines(statistics.mean(delta_losses["Top3"][95:100]),
               xmin=frame_x_min, xmax=100, label="Mean Top3",
               colors="darkgray", linestyles="--")
    delta_losses["Top4"].plot.line(color="lightgray")
    plt.hlines(statistics.mean(delta_losses["Top4"][95:100]),
               xmin=frame_x_min, xmax=100, label="Mean Top4",
               colors="lightgray", linestyles="--")
    plt.show()
    fig_small.savefig(f'{folder}deltalosses_small.png', bbox_inches="tight")

    print_done()
