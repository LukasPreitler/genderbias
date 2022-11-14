import os
import statistics
from more_itertools import random_permutation

import pandas as pd
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

from colorama import Fore
from colorama import Style

def print_hi(name):
    print(f'{Fore.GREEN}------------------------------- Start, {name} --------------------------------------------------{Style.RESET_ALL}')  # Press Strg+F8 to toggle the breakpoint.

def print_step(step):
    print(f'{Fore.GREEN}------------------------------- Next step is {step} ---------------------------------------------{Style.RESET_ALL}')

def print_inter_step(inter_step):
    print(f'{Fore.YELLOW}------------------------------- {inter_step} ---------------------------------------------{Style.RESET_ALL}')

def print_done():
    print(f'{Fore.GREEN}------------------------------- Script is finished ---------------------------------------------{Style.RESET_ALL}')

def print_permutationprocess(step):
    if step % 100 == 0:
        print("#", end="")


def permutationTest(testgrupp, controllgrupp, iter):
    ulist = testgrupp + controllgrupp

    test_value = statistics.mean(testgrupp) - statistics.mean(controllgrupp)
    store = []
    for t in range(iter):
        plist = (random_permutation(ulist, len(ulist)))
        x = plist[len(testgrupp):]
        y = plist[:len(controllgrupp)]
        store.append(statistics.mean(x) - statistics.mean(y))

    count = 0
    for t in store:
        if t < test_value:
            count = count + 1
    p_value = count/len(store)
    return p_value

if __name__ == '__main__':
    print_hi('evaluted hyperparameter')

    folder = "model/random/losses/"
    filename = "best_models"
    save_folder = "model/random/permutationModel/"
    os.makedirs(f"model/random/losses/tables", exist_ok=True)

    model_table = pd.read_csv(f'{folder}delta_losses.csv')

    formatter = {"epochs": "{:d}",
                 "min_word": "{:d}",
                 "window": "{:d}",
                 "vector": "{:d}",
                 "negSample": "{:d}",
                 "losses": "{:,8.2f}",
                 }

    mod_model_table = model_table.loc[(model_table['losses'] != 0),
                                      ["epochs", "min_word", "window", "vector",
                               "negSample", "losses"]
                              ]

    mod_model_table = mod_model_table.nsmallest(len(mod_model_table), "losses")
    print(mod_model_table)
    print_inter_step(f"Model table has {len(mod_model_table)} rows")

    mod_model_table.to_csv(f"{folder}{filename}.csv")
    mod_model_table.style.format({"epochs": '{:.0f}',
                              "min_word":  '{:.0f}',
                              "window":  '{:.0f}',
                              "vector":  '{:.0f}',
                              "negSample":  '{:.0f}'},
                                 decimal=',',
                                 precision=3,
                                 thousands='.').to_latex(f"{folder}/tables/{filename}.tex")

    print_inter_step("check for statistic signification")

    minWordVec = [1, 2, 3, 4, 5]
    test_epochs = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # len() = 7
    windowSizeVec = [1, 2, 3, 5, 8, 13]  # len() = 6
    vsVec = [25, 50, 100, 200, 300, 400, 500]  # len() = 7
    negVec = [1, 2, 3, 5, 8, 13]  # len() = 6

    generated_models_performence = mod_model_table

    print_inter_step("Min-word")
    min_p_value = pd.DataFrame(columns=minWordVec,
                               index=minWordVec)

    for i in minWordVec:
        for j in minWordVec:
            if i != j:
                tmpI = generated_models_performence.loc[(generated_models_performence["min_word"] == i), ["losses"]].values
                tmpJ = generated_models_performence.loc[(generated_models_performence["min_word"] == j), ["losses"]].values
                tmpI = [item for sublist in tmpI.tolist() for item in sublist]
                tmpJ = [item for sublist in tmpJ.tolist() for item in sublist]

                p_value = permutationTest(tmpI, tmpJ, 1000)
                min_p_value.at[i, j] = p_value
    min_p_value = pd.concat([min_p_value, min_p_value.mean(axis=1)], axis=1)
    print(min_p_value)

    min_p_value.to_csv(f"{save_folder}minW.csv")
    min_p_value.style.format(decimal=',', precision=3).to_latex(f"{folder}/tables/minW.tex")

    print_inter_step("Epochs")
    epochs_table = pd.DataFrame(columns=test_epochs,
                                index=test_epochs)
    for i in test_epochs:
        for j in test_epochs:
            if j != i:
                tmpI = generated_models_performence.loc[(generated_models_performence["epochs"] == i), ["losses"]].values
                tmpJ = generated_models_performence.loc[(generated_models_performence["epochs"] == j), ["losses"]].values
                tmpI = [item for sublist in tmpI.tolist() for item in sublist]
                tmpJ = [item for sublist in tmpJ.tolist() for item in sublist]
                p_value = permutationTest(tmpI, tmpJ, 1000)
                epochs_table.at[i, j] = p_value

    epochs_table = pd.concat([epochs_table, epochs_table.mean(axis=1)], axis=1)
    print(epochs_table)

    epochs_table.to_csv(f"{save_folder}epoch.csv")
    epochs_table.style.format(decimal=',', precision=3).to_latex(f"{folder}/tables/epoch.tex")

    print_inter_step("WindowSize")
    windowSizeVec_table = pd.DataFrame(columns=windowSizeVec,
                                       index=windowSizeVec)

    for i in windowSizeVec:
        for j in windowSizeVec:
            if i != j:
                tmpI = generated_models_performence.loc[(generated_models_performence["window"] == i), ["losses"]].values
                tmpJ = generated_models_performence.loc[(generated_models_performence["window"] == j), ["losses"]].values
                tmpI = [item for sublist in tmpI.tolist() for item in sublist]
                tmpJ = [item for sublist in tmpJ.tolist() for item in sublist]

                p_value = permutationTest(tmpI, tmpJ, 1000)
                windowSizeVec_table.at[i, j] = p_value
    windowSizeVec_table = pd.concat([windowSizeVec_table, windowSizeVec_table.mean(axis=1)], axis=1)
    print(windowSizeVec_table)

    windowSizeVec_table.to_csv(f"{save_folder}WS.csv")
    windowSizeVec_table.style.format(decimal=',', precision=3).to_latex(f"{folder}/tables/WS.tex")

    print_inter_step("Vector Size")
    table_vsVec = pd.DataFrame(columns=vsVec,
                               index=vsVec)

    for i, size_i in enumerate(vsVec):
        for j, size_j in enumerate(vsVec):
            if i != j:
                tmpI = generated_models_performence.loc[(generated_models_performence["vector"] == vsVec[i]), ["losses"]].values
                tmpJ = generated_models_performence.loc[(generated_models_performence["vector"] == vsVec[j]), ["losses"]].values
                tmpI = [item for sublist in tmpI.tolist() for item in sublist]
                tmpJ = [item for sublist in tmpJ.tolist() for item in sublist]

                p_value = permutationTest(tmpI, tmpJ, 1000)
                table_vsVec.at[vsVec[i], vsVec[j]] = p_value
    table_vsVec = pd.concat([table_vsVec, table_vsVec.mean(axis=1)], axis=1)
    print(table_vsVec)

    table_vsVec.to_csv(f"{save_folder}VS.csv")
    table_vsVec.style.format(decimal=',', precision=3).to_latex(f"{folder}/tables/VS.tex")

    print_inter_step("Neg Sample")
    table_negVec = pd.DataFrame(columns=negVec,
                                index=negVec)

    for i in negVec:
        for j in negVec:
            if i != j:
                tmpI = generated_models_performence.loc[(generated_models_performence["negSample"] == i), ["losses"]].values
                tmpJ = generated_models_performence.loc[(generated_models_performence["negSample"] == j), ["losses"]].values
                tmpI = [item for sublist in tmpI.tolist() for item in sublist]
                tmpJ = [item for sublist in tmpJ.tolist() for item in sublist]
                if len(tmpI) == 0 or len(tmpJ) == 0:
                    continue

                p_value = permutationTest(tmpI, tmpJ, 1000)
                table_negVec.at[i, j] = p_value
    table_negVec = pd.concat([table_negVec, table_negVec.mean(axis=1)], axis=1)
    print(table_negVec)

    table_negVec.to_csv(f"{save_folder}negS.csv")
    table_negVec.style.format(decimal=',', precision=3).to_latex(f"{folder}/tables/negS.tex")
    print_done()
