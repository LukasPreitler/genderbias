import math
import random

import statistics
import inline as inline
import matplotlib
import pandas as pd
from colorama import Fore
from colorama import Style

pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

from more_itertools import random_permutation, distribute

from mlxtend.evaluate import permutation_test
import numpy as np

def print_hi(name):
    print(f'{Fore.GREEN}------------------------------- Start, {name} --------------------------------------------------{Style.RESET_ALL}')  # Press Strg+F8 to toggle the breakpoint.

def print_step(step):
    print(f'{Fore.GREEN}------------------------------- Next step is {step} ---------------------------------------------{Style.RESET_ALL}')

def print_done():
    print(f'{Fore.GREEN}------------------------------- Script is finished ---------------------------------------------{Style.RESET_ALL}')

def print_progress(iter):
    if iter % 1000 == 0 and iter != 0:
        print(f'{Fore.YELLOW}#{Style.RESET_ALL}', end="")
    #elif iter == 1:
        #print(f'{Fore.YELLOW}#{Style.RESET_ALL}', end="")

def print_top_sim(word, topNumber):
    print(f"{word} is most similare to: ")
    store = text_model.wv.most_similar_cosmul(word, topn=topNumber)
    for sim in store:
        print(sim)

def print_cosim(list):
    wordlist = []
    femalelist = []
    malelist = []
    for i in list:
        try:
            cos_female = text_model.wv.similarity("female_name", i)
            cos_male = text_model.wv.similarity("male_name", i)
            wordlist.append(i)
            femalelist.append(cos_female)
            malelist.append(cos_male)
        except:
            print(f"{i} is not in the corpura")

    d = {"female names": femalelist, "male names": malelist}
    df = pd.DataFrame(data=d, index=wordlist)
    print(df)


def build_s(listW, listA, listB):
    listResult = []
    for x in listW:
        w_a = []
        w_b = []
        for a in listA:
            w_a.append(text_model.wv.similarity(x, a))
        for b in listB:
            w_b.append(text_model.wv.similarity(x, b))
        listResult.append(statistics.mean(w_a) - statistics.mean(w_b))
    return listResult

def category(listX, listY, listM, listF, permutations=50000):
    print_step("New X/Y - Test")
    print(f"list X: {listX}")
    print(f"list Y: {listY}")
    print(f"list A: {listM}")
    print(f"list B: {listF}")
    if not len(listX) or not len(listY):
        print("no permutation test possible")
        return None, None, None, None
    if len(listX) < 4 or len(listY) < 4:
        print(f"{Fore.YELLOW} Beware {Style.RESET_ALL}")
    s_x = build_s(listX, listM, listF)
    s_y = build_s(listY, listM, listF)

    m_s = statistics.mean(s_x) - statistics.mean(s_y)

    # cohens_d
    s_top = (statistics.mean(s_x) - statistics.mean(s_y))
    s_w = build_s(listX + listY, listM, listF)
    s_w_mean = statistics.mean(s_w)
    s_botom = 0
    for elem in s_w:
        s_botom = s_botom + pow(elem - s_w_mean, 2)
    s_botom = math.sqrt(s_botom/(len(s_w)))
    m_cohens = s_top/s_botom

    s_x = build_s(listX, listF, listM)
    s_y = build_s(listY, listF, listM)

    f_s = statistics.mean(s_x) - statistics.mean(s_y)

    s_top = (statistics.mean(s_x) - statistics.mean(s_y))
    s_w = build_s(listX + listY, listM, listF)
    s_w_mean = statistics.mean(s_w)
    s_botom = 0
    for elem in s_w:
        s_botom = s_botom + pow(elem - s_w_mean, 2)
    s_botom = math.sqrt(s_botom/(len(s_w)))
    f_cohens = s_top/s_botom

    f_p_value, m_p_value = permutationtest(m_s, f_s, listX, listY, listF, listM, permutations)

    print(f'female effect size = {m_cohens}')
    print(f"male effect size = {f_cohens}")

    return f_p_value, f_cohens, m_p_value, m_cohens

def prepareList(Words, model):
    doRemove = []
    for index, word in enumerate(Words):
        if tagger.analyze(word)[0] != "":
            Words[index] = tagger.analyze(word)[0]

        if not model.wv.has_index_for(Words[index]):
           doRemove.append(Words[index])

    for word in doRemove:
        Words.remove(word)

def prepareDict(Dict_org, model):
    for element in Dict_org:
        prepareList(Dict_org[element], model)

def permutationtest(test_stat_m, test_stat_f, listX, listY, listF, listM, permutations=10000):
    uList = listX + listY
    #print(f"lenght of ulist: {len(uList)} {uList}")

    tstat_m_x = []
    tstat_f_x = []
    #print_step(f"Calculate the permutation list size = {len(uList)}")

    for iteration in range(0, permutations):
        print_progress(iteration)
        plist = (random_permutation(uList, len(uList)))
        x = plist[len(listX):]
        y = plist[:len(listX)]

        m_s_x = build_s(list(x), listM, listF)
        m_s_y = build_s(list(y), listM, listF)

        f_s_x = build_s(list(x), listF, listM)
        f_s_y = build_s(list(y), listF, listM)

        tstat_m_x.append(statistics.mean(m_s_x) - statistics.mean(m_s_y))
        tstat_f_x.append(statistics.mean(f_s_x) - statistics.mean(f_s_y))

    m_count_greater = 0
    f_count_greater = 0
    for test in tstat_m_x:
        if test >= test_stat_m:
            m_count_greater = m_count_greater + 1

    for test in tstat_f_x:
        if test >= test_stat_f:
            f_count_greater = f_count_greater + 1

    print(f"\nTest statistik: female={test_stat_f} observed \n"
          f"A=female p-value: {f_count_greater/len(tstat_f_x)} = {f_count_greater}/{len(tstat_f_x)}"
          f"\nTest statistik: male={test_stat_m} observed \n"
          f"A=male p-value: {m_count_greater / len(tstat_f_x)} = {m_count_greater}/{len(tstat_m_x)}"
          )

    #print(tstat)
    return f_count_greater / len(tstat_f_x), m_count_greater / len(tstat_f_x)  # p-value


def analog(model, wrd1, wrd2, wrd3):
    store = model.wv.most_similar(negative=[wrd1],
                                   positive=[wrd2, wrd3])
    return store[0][0]
# towardsdatascience.com/how-to-solve-analogies-with-word2vec-6ebaf2354009


if __name__ == '__main__':
    print_hi("Analyse")

    print_step("load model")
    from gensim import models

    load_folder = "model/random/final/"

    print_step("load HanoverTagger for lammaziation")
    from HanTa import HanoverTagger as ht
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')

    print_step("load model")

    text_model = models.Word2Vec.load(f"{load_folder}model")
    print(text_model)

    print_step("load gender list")
    gender_list = {
        # Gender Words
        "female_list": ['Tochter', 'Mutter', 'Frau', 'weiblich', 'Schwester', 'mama', 'Tante', 'female_name'],
        "male_list": ['Sohn', 'Vater', 'Mann', 'männlich', 'Bruder', 'Papa', 'Onkel', 'male_name'],
    }

    print_step("x/y-lists")
    x_y_lists = {

        # 1
        # weat 5: pleasant - unpleasent
        "pleasant": ["Spass", "Liebe", "Frieden", "wunderbar", "Freude", "Lachen", "glücklich"],
        "unpleasent": ["Qual", "furchtbar", "schrecklich", "übel", "böse", "Krieg", "scheusslich", "Versagen"],

        # 2
        # weat 6: career - Family
        "career": ['Geschäftsführung', 'Management', 'professionell', 'Unternehmen', 'Gehalt', 'Büro', 'Geschäft',
                   'Karriere'],
        "family": ['Zuhause', 'Eltern', 'Kinder', 'Familie', 'Cousins', 'Heirat', 'Hochzeit', 'Verwandte'],

        # 3
        # weat 7: math - arts
        "math": ["Mathematik", "Algebra", "Geometrie", "Calculus", "Gleichungen", "Berechnung", "Zahlen", "Addition"],
        "arts": ["Poesie", "Kunst", "Tanz", "Literatur", "Roman", "Symphonie", "Drama", "Skulptur"],

        # 4
        # weat 8: science - arts
        "science": ["Wissenschaft,Technologie", "Physik", "Chemie", "Einstein", "NASA", "Experiment", "Astronomie"],
        "arts2": ["Poesie", "Kunst", "Shakespeare", "Tanz", "Literatur", "Roman", "Symphonie", "Drama"],

        # 5
        # character:
        "ratio": ["Geist", "Vernunft", "Verstand", "Denken", "Wissen", "Urteilen"],
        "passion": ["Gefühl", "Empfinden", "Empfänglichkeit", "Rezeptivität", "Religiosität", "Verstehen"],

        # 6
        # W4: intelligent - appearance
        "intelligent": ["frühreif", "einfallsreich", "wissbegierig", "genial", "erfinderisch", "scharfsinnig",
                        "anpassungsfähig", "reflektiert", "einfühlsam", "intuitiv", "neugierig", "umsichtig",
                        "analytisch", "treffend", "ehrwürdig", "einfallsreich", "scharfsinnig", "aufmerksam", "weise",
                        "klug", "raffiniert", "clever", "brillant", "logisch", "intelligent"],
        "appearance": ["verführerisch", "üppig", "errötend", "häuslich", "mollig", "sinnlich", "hinreißend", "schlank",
                       "kahl", "sportlich", "modisch", "gedrungen", "hässlich", "muskulös", "schmächtig", "schwächlich",
                       "ansehnlich", "gesund", "attraktiv", "fett", "schwach", "dünn", "hübsch", "schön", "stark"],

        # 7
        # W5: strength - weakness
        "strength": ["Macht", "stark", "selbstbewusst", "dominant", "kraftvoll", "Befehl", "durchsetzen", "laut",
                     "kühn",
                     "gelingen", "triumphieren", "anführen", "schreien", "dynamisch", "gewinnen"],
        "weakness": ["schwach", "kapitulieren", "ängstlich", "verletzlich", "Schwäche", "wischiwaschi", "zurückziehen",
                     "nachgeben", "versagen", "schüchtern", "folgen", "verlieren", "zerbrechlich", "ängstlich",
                     "Verlierer"],

        # 8
        # W6: outdoor - indoor
        "outdoor": ["draußen", "außen", "Natur", "Garten", "Baum", "Hinterhof", "See", "Berg"],
        "indoor": ["innen", "drinnen", "Küche", "Haushalt", "Zuhause", "Sofa", "schlafzimmer", "Badezimmer"],

        # 9
        # W7: male toys vs female toys
        "male_toys": ["Ball", "Schläger", "Lkw", "Auto", "Fahrrad", "Pistole", "Soldat", "blau"],
        "female_toys": ["Puppe", "Puppenhaus", "barbie", "Schminke", "Ballerina", "schmuck", "Pony", "rosa"],

        # 10
        # w8: male vs female sports
        "male_sports": ["Football", "Basketball", "Baseball", "Fußball", "Rugby", "Boxen", "Fahrradfahren"],
        "female_sports": ["Volleyball", "Turnen", "Netzball", "Softball", "Cheerleader", "Tanzen", "Schlittschuhlaufen",
                          "Lacrosse"],

        # 11
        # w9:
        "active_games": ["fliegt", "fährt", "springt", "klettert", "schwimmt", "rutscht", "taucht", "hüpft"],

        "silent_games": ["liest", "schreibt", "beobachtet", "versteckt", "hört zu", "zeichnet", "malt", "skizziert"],

        # 12
        # w10:
        "male_adjectives": ["rebellisch", "begabt", "störrisch", "eigenwillig", "charismatisch", "zynisch", "mürrisch",
                            "naiv"],
        "female_adjectives": ["qualifiziert", "verehrt", "geachtet", "einflussreich", "professionell", "bekannt",
                              "begabt", "erfolgreich"],

        # 13
        # w11:
        "male_professions": ["Klempner", "Klempnerin", "Mechaniker", "Mechanikerin", "Schreiner", "Schreinerin",
                             "Maschinist", "Maschinistin", "Ingenieur", "Ingenieurin", "Programmierer",
                             "Programmiererin",
                             "Architekt", "Architektin", "Offizier", "Offizierin"],

        "female_professions": ["Hygieniker", "Hygienikerin", "Friseur", "Friseurin", "Krankenschwester",
                               "Krankenpfleger",
                               "Bibliothekar", "Bibliothekarin", "Planer", "Planerin", "Therapeut", "Therapeutin",
                               "Praktiker", "Praktikerin", "Lehrer", "Lehrerin"],

        # 14
        # w12:
        "male_qualities": ["kompetent", "instrumental", "erfolgreich", "motiviert", "klug", "Abenteuer", "verdienen",
                           "meistern"],
        "female_qualities": ["pflegend", "abhängig", "gehorsam", "inkompetent", "passiv", "Opfer", "erfolglos",
                             "unsichtbar"],

        # 15
        # w13:
        "dominant": ["dominant", "herrschend", "unterdrückend", "kontrollierend", "befehlend", "überlegen", "Autorität",
                     "instrumental"],
        "obedient": ["gehorsam", "willig", "aufmerksam", "rücksichtsvoll", "wohlerzogen", "höflich", "gezwungen",
                     "kooperativ"],

        # 16
        # w14:
        "male_subjects": ["Mathematik", "Physik", "Wissenschaft", "Chemie", "Informatik", "Ingenieurswesen", "Sport",
                          "Technik"],

        "female_subjects": ["Geisteswissenschaften", "Kunst", "Bildung", "Biologie", "Medizin", "Sprache", "Englisch",
                            "Musik"]
    }

    key_x = ["pleasant", "career", "math", "science",
             "ratio", "intelligent", "strength", "outdoor",
             "male_toys", "male_sports", "active_games", "male_adjectives",
             "male_professions", "male_qualities", "dominant", "male_subjects"]
    key_y = ["unpleasent", "family", "arts", "arts2",
             "passion", "appearance", "weakness", "indoor",
             "female_toys", "female_sports", "silent_games", "female_adjectives",
             "female_professions", "female_qualities", "obedient", "female_subjects"]

    index = ["pleasant vs unpleasent",
             "career vs family",
             "math vs arts",
             "science vs arts2",
             "ratio vs passion",
             "intelligent vs appearance",
             "strength vs weakness",
             "outdoor vs indoor",
             "male_toys vs female_toys",
             "male_sports vs female_sports",
             "active_games vs silent_games",
             "male_adjectives vs female_adjectives",
             "male_professions vs female_professions",
             "male_qualities vs female_qualities",
             "dominant vs obedient",
             "male_adjectives vs female_subjects"]

    prepareDict(gender_list, text_model)
    gender_list["female_list"].append("female_name")
    gender_list["male_list"].append("male_name")
    prepareDict(x_y_lists, text_model)

    store = pd.DataFrame(columns=["category", "female p-value", "cohens'd", "male p-value"])

    for i in range(len(key_x)):
        f_p_value, cohens, m_p_value, m_cohens = category(x_y_lists[key_x[i]],
                                                          x_y_lists[key_y[i]],
                                                          gender_list["male_list"],
                                                          gender_list["female_list"], 50000)
        data = [{"category": f"{key_x[i]} vs {key_y[i]}",
                 "female p-value": f_p_value,
                 "cohens'd": cohens,
                 "male p-value": m_p_value}]
        store = store.append(data, ignore_index=True)

    store.to_csv(f"{load_folder}weat.csv", float_format='%.3f')

print_done()
