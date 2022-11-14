import random
import statistics
import numpy as np

import pandas as pd
import pickle

from colorama import Fore
from colorama import Style

from time import time, monotonic
import os
from gensim.models.callbacks import CallbackAny2Vec

from HanTa import HanoverTagger as ht
import seaborn as sns
sns.set_style("darkgrid")


def print_hi(name):
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

def preparePickList(arg_list, mul):
    new_list = []
    for t in range(mul):
        new_list = new_list + arg_list
    return new_list

# init callback class
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
    print_hi('generate models')

    print_step("load HanoverTagger for lammaziation")
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')

    print_step("Build model")
    # source: https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook#Gensim-Word2Vec-Implementation:
    import multiprocessing
    from gensim.models import Word2Vec

    cores = multiprocessing.cpu_count()

    print_step("load text")

    folder = "model/random/losses/"
    filename = "delta_losses"

    total_models = 0
    hyperparameter = pd.DataFrame()
    os.makedirs(f'{folder}', exist_ok=True)

    with open(f'corpora/cleaned_text.pickle', 'rb') as f:
        sentences_token_cleaned = pickle.load(f)

    print_step("inital model")

    store = pd.DataFrame(columns=["min_word", "window", "vector", "negSample",
                                  "weat_count", "cohens-score", "p-value-score"])

    modNum = 0
    total_models = 210 # todo: change your number of generated models for your need
    data = []
    minWordVec = []
    minWordpicVec = []

    # todo: change minium word count with respect to your preprocess
    minWordVec = [1, 2, 3, 4, 5]
    minWordpicVec = preparePickList(minWordVec, 42)  # len() = 5

    # todo: change the value if needed
    test_epochs = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # len() = 7
    windowSizeVec = [1, 2, 3, 5, 8, 13]  # len() = 6
    vsVec = [25, 50, 100, 200, 300, 400, 500]  # len() = 7
    negVec = [1, 2, 3, 5, 8, 13]  # len() = 6

    # todo: change the value if needed
    picVec_epochs = preparePickList(test_epochs, 21)
    windowSizepicVec = preparePickList(windowSizeVec, 35)
    vsPicVec = preparePickList(vsVec, 30)
    negPicVec = preparePickList(negVec, 35)

    for test_model in range(total_models):

        epochs = random.choice(picVec_epochs)
        picVec_epochs.remove(epochs)

        minWord = random.choice(minWordpicVec)
        minWordpicVec.remove(minWord)

        windowSize = random.choice(windowSizepicVec)
        windowSizepicVec.remove(windowSize)

        vs = random.choice(vsPicVec)
        vsPicVec.remove(vs)

        neg = random.choice(negPicVec)
        negPicVec.remove(neg)

        print_step(f"Neue Model {test_model + 1}/{total_models}: "
                   f"epochs={epochs}, min={minWord}, window={windowSize}, "
                   f"vector={vs}, neg={neg}")
        table = []
        epoch_logger = EpochLogger(table)
        textmodel = Word2Vec(min_count=minWord,
                             window=windowSize,
                             vector_size=vs,
                             negative=neg,
                             workers=cores - 1
                             )

        corpura_count = 0
        for sent in sentences_token_cleaned:
            for word in sent:
                corpura_count = corpura_count + 1

        print_inter_step(f"build vocab from {corpura_count} words")

        t = time()

        textmodel.build_vocab(sentences_token_cleaned, progress_per=1000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        textmodel.train(sentences_token_cleaned,
                        total_examples=textmodel.corpus_count,
                        epochs=epochs,
                        report_delay=1,
                        compute_loss=True,
                        callbacks=[epoch_logger])
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

        print_inter_step(f"validate")

        losses = statistics.mean(table[-5:])

        elements = [epochs, minWord, windowSize, vs, neg,
                    losses]
        data.append(elements)

        to_save = pd.DataFrame(data=np.array(data),
                               columns=["epochs", "min_word", "window", "vector",
                                        "negSample", "losses"])

        to_save.to_csv(f'{folder}{filename}.csv')
        print(end="\n")

        print_inter_step("Save Hyperparameter:"
                         f"epochs={epochs}"
                         f"min_count={minWord}, "
                         f"window={windowSize}, "
                         f"vector_size={vs}, "
                         f"negative={neg}."
                         f"\n target values:"
                         f"losses ={losses}")
    print_done()
