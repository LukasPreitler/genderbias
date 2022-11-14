import pandas as pd
from colorama import Fore
from colorama import Style
from time import time
import pickle
import codecs
import nltk
from HanTa import HanoverTagger as ht
import re
import matplotlib.pyplot as plt

def print_hi(name):
    print(f'{Fore.GREEN}------------------------------- Start, {name} --------------------------------------------------{Style.RESET_ALL}')

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


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    print_hi('preprocessing')

    # open text
    fd = codecs.open("text.txt", "r", "utf-8")
    # open result file
    fd_results = open(f"results.txt", "w")

    print_step(f'load extracted texts')
    texts = fd.read()
    fd.close()

    print_step('tokenize')
    # source: https://textmining.wp.hs-hannover.de/Preprocessing.html
    sentences = nltk.sent_tokenize(texts, language='german')
    t = time()
    sentences_token = [nltk.tokenize.word_tokenize(sent, language='german')
                        for sent in sentences]

    print_step("Prepare the token")
    # source: https://github.com/wartaal/HanTa/blob/master/Demo.ipynb

    print_inter_step("load HanoverTagger for lammaziation")
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')

    print_inter_step("load female and male names lists")
    fd_female_name = open('lists/names_female.txt')
    female_names = fd_female_name.read().split()
    fd_female_name.close()

    # remove names with like other meaning
    print(f'load {len(female_names)} female names', end=" ")
    fd_female_name_meaning = open('lists/female_names_meaning.txt')
    names_female_meaning = fd_female_name_meaning.read().split()
    female_names = [word for word in female_names if word not in names_female_meaning]
    print(f' {len(female_names)} after remove {names_female_meaning}')

    fd_male_name = open('gender_namelists/names_male.txt')
    male_names = fd_male_name.read().split()
    fd_male_name.close()

    print(f'load {len(male_names)} male names', end=" ")
    fd_names_male_meaning = open('lists/male_names_meaning.txt.txt')
    names_male_meaning = fd_names_male_meaning.read().split()
    male_names = [word for word in male_names if word not in names_male_meaning]
    print(f' {len(male_names)} after remove {names_male_meaning}  ')

    #check for gender neutral names
    print_inter_step(f'check if names female and male')
    gender_neutral_names = []
    word_index_male = 0
    word_index_female = 0
    letter_index = 0

    female_names = sorted(female_names)
    male_names = sorted(male_names)

    while word_index_female < len(female_names) and word_index_male < len(male_names):
        print("#", end="")
        if letter_index + 1 > len(female_names[word_index_female]) or letter_index + 1 > len(
                male_names[word_index_male]):
            if len(female_names[word_index_female]) < len(male_names[word_index_male]):
                word_index_female += 1
                letter_index = 0
            else:
                word_index_male += 1
                letter_index = 0

        if female_names[word_index_female][letter_index] == male_names[word_index_male][letter_index]:
            if len(female_names[word_index_female]) == letter_index + 1:
                gender_neutral_names.append(female_names[word_index_female])
                female_names.remove(female_names[word_index_female])
                male_names.remove(male_names[word_index_male])
                letter_index = 0
            else:
                letter_index += 1
        else:
            if ord(female_names[word_index_female][letter_index]) < ord(male_names[word_index_male][letter_index]):
                word_index_female += 1
                letter_index = 0
            else:
                word_index_male += 1
                letter_index = 0

    print("")
    print(f'Neutral names: {gender_neutral_names}')

    n_female = ['Ada', 'Aida', 'Alessa', 'Alexa', 'Alla', 'Alva', 'Amira', 'Ana', 'Ann', 'Antonie', 'Aurelia', 'Bea',
                'Brenda', 'Cami', 'Caro', 'Cassia', 'Dana', 'Daria', 'Devi', 'Emilia', 'Eva', 'Eve', 'Evi', 'Fabia',
                'Fabienne', 'Filiz', 'Finja', 'Finnja', 'Floria', 'Friede', 'Fritzi', 'Geri', 'Gill', 'Hana', 'Hilde',
                'Imilia', 'Io', 'Irene', 'Iva', 'Jane', 'Jeanne', 'Jessy', 'Jola', 'Josi', 'Joy', 'Jule', 'Julie',
                'Kaimana', 'Karli', 'Kea', 'Keona', 'Kira', 'Kiri', 'Kora', 'Lauren', 'Lea', 'Leia', 'Lena', 'Leona',
                'Leoni', 'Lia', 'Lil', 'Linn', 'Lua', 'Lucia', 'Mali', 'Malu', 'Maria', 'Mila', 'Noa', 'Noelani', 'Nola',
                'Norma', 'Octavia', 'Oda', 'Olive', 'Orla', 'Osma', 'Pam', 'Pippi', 'Ragna', 'Rena', 'Resa', 'Rica',
                'Romy', 'Sara', 'Sila', 'Sina', 'Siri', 'Thora', 'Tia', 'Tizia', 'Tora', 'Uma', 'Una', 'Valeria',
                'Vida', 'Wilma', 'Xavia', 'Xia', 'Xuan', 'Yannie', 'Zlata', 'Zoe', 'Zora']

    n_male = ['Adrian', 'Andy', 'Callisto', 'Corin', 'Etienne', 'Frana', 'Gabriel', 'Glenn','Gustave', 'Harper', 'Hauke',
              'Ike', 'Jona', 'Jonah', 'Juri', 'Kai', 'Lean', 'Lenny', 'Leo', 'Lucian', 'Makani', 'Manuel', 'Marin',
              'Nalu', 'Noel', 'Ola', 'Quoc', 'Thore', 'Tobia']

    for name in n_male:
        male_names.append(name)
        gender_neutral_names.remove(name)

    for name in n_female:
        female_names.append(name)
        gender_neutral_names.remove(name)

    gender_neutral_names.remove("Juli")
    gender_neutral_names.remove("Bo")

    print(f'Neutral names: {gender_neutral_names}')
    print(f"Count female {word_index_female}/{len(female_names)}, count male {word_index_male}/{len(male_names)}")
    print(f'Count gender neutral names: {len(gender_neutral_names)}')
    print_inter_step(f'load {len(gender_neutral_names)} gender_neutral names')

    print_inter_step('load stoplist')
    # source: https://www.kaggle.com/rtatman/stopword-lists-for-19-languages?select=germanST.txt

    f_stoplist = open('lists/stoplist.txt', 'r')
    stoplist = f_stoplist.read()
    f_stoplist.close()
    # print(stoplist)

    print_inter_step("Prepare Artefakt detection")
    print_step("Replace token with gender_token, remove stopwords and artefacts, ...")
    female_count = 0
    male_count = 0
    neutral_count = 0
    sie_count = 0
    er_count = 0

    female_name = []
    male_name = []
    neutral_name = []

    count_stoplist = 0
    artefact_count = 0
    count_sent = 0
    count_word = 0

    sentences_token_cleaned = []

    for sent in sentences_token:
        count_sent = count_sent + 1
        #print(sent)
        del_list = []
        for index, word in enumerate(sent):
            print_progress(count_word)
            count_word = count_word + 1

            if len(word) < 2 or re.match("cid:", word) is not None:
                #print(f'art={word}', end=", ")
                del_list.append(index)
                artefact_count += 1

            elif word.lower() in stoplist:
                #print(f'stop={word}', end=", ")
                count_stoplist = count_stoplist + 1
                del_list.append(index)

            elif word in female_names:
                #print(f'fem={word}', end=", ")
                female_name.append(word)
                sent[index] = 'female_name'
                female_count += 1

            elif word in male_names:
                #print(f'male={word}', end=", ")
                male_name.append(word)
                sent[index] = 'male_name'
                male_count += 1

            elif word in gender_neutral_names:
                #print(f'neutral={word}', end=", ")
                neutral_name.append(word)
                sent[index] = 'neutral_name' + str(gender_neutral_names)
                neutral_count += 1
            else:
                if tagger.analyze(word)[0] != "":
                    sent[index] = tagger.analyze(word)[0]

        for elem in del_list[::-1]:
            #print(f'del: {sent[elem]}')
            del sent[elem]
        #print(f'add sent: {sent}')
        sentences_token_cleaned.append(sent)

    print("")
    print(f'{count_sent} number of Sentance  with {count_word} words checkt')
    print(f'{female_count} female names replaced, {male_count} male names replaces and {neutral_count} neutral name are replaced')
    print(f'{count_stoplist} stop list word and {artefact_count} artifacts removed')

    fd_results.write(f'\n{count_sent} number of Sentance  with {count_word} words checkt\n '
                     f'{female_count} female names replaced, {male_count} male names replaces and {neutral_count} neutral name are replaced \n'
                     f'{count_stoplist} stop list word and {artefact_count} artifacts removed \n')

    word_count = 0
    for sent in sentences_token_cleaned:
        for word in sent:
            word_count = word_count + 1

    print(f'from {count_word} to {word_count}')
    #print(f'sentances cleaned: {sentences_token_cleaned}')
    print(f'neutral names: {neutral_name}')
    print(f'female names: {female_name}')
    print(f'male names: {male_name}')

    min_appearance = 0
    print_step(f'remove words that appearence is under the threshold of {min_appearance}')

    print_inter_step('start count appearence of token')
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in sentences_token_cleaned:
        for token in text:
            frequency[token] += 1

    print_inter_step(f'remove token is under the threashold of {min_appearance}')
    corpura_sent = []

    min_word_count = 0
    for sent in sentences_token_cleaned:
        new_sent = []
        for token in sent:
            if frequency[token] > min_appearance:
                new_sent.append(token)
            else:
                min_word_count = min_word_count + 1

        corpura_sent.append(new_sent)


    print(f'min words removed: {min_word_count}')

    print_inter_step('Print frequence of most common words')
    fdist = nltk.FreqDist(frequency)

    frequency_neutral = defaultdict(int)
    for name in neutral_name:
        frequency_neutral[name] += 1
    frequency_female = defaultdict(int)
    for name in female_name:
        frequency_female[name] += 1
    frequency_male = defaultdict(int)
    for name in male_name:
        frequency_male[name] += 1

    freq_list = [frequency_neutral, frequency_female, frequency_male]

    print(f"length of dict neutral: "
          f"{len(frequency_neutral)}, "
          f"female: {len(frequency_female)} "
          f"male: {len(frequency_male)}")

    fd_results.write("\n length of dict neutral:\n "
          f"{len(frequency_neutral)}, \n"
          f"female: {len(frequency_female)} \n"
          f"male: {len(frequency_male)}\n")

    round = 0
    for list in freq_list:
        fig = plt.figure(figsize=(10, 4))
        plt.gcf().subplots_adjust(bottom=0.15)  # to avoid x-ticks cut-off
        fdist = nltk.FreqDist(list)
        fdist.plot(30, cumulative=False, show=False, color="black")
        plt.ylabel("Anzahl")
        plt.xlabel("Vornamen")

        name = f'{list=}'.split('=')[0]
        fig.savefig(f'pic/text_nameFreq{round}.png', bbox_inches="tight")
        round = round + 1
        print(f"round {round}")
        df = pd.DataFrame.from_dict(list.items())
        with open(f'results/nameFreq_text_{round}.txt', 'w') as tf:
            tf.write(df.to_latex())

    print_step(f"Save cleaned text")
    f_cleaned = open(f'corpora/cleaned_text.pickle', 'wb')
    pickle.dump(sentences_token_cleaned, f_cleaned)
    f_cleaned.close()

print_done()