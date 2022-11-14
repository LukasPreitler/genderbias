import pandas as pd
from colorama import Fore
from colorama import Style
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import umap
import umap.plot

from gensim import models

import numpy as np
import pandas as pd
from typing import Dict
import pickle as pkl
from collections import namedtuple, OrderedDict
import tqdm

from HanTa import HanoverTagger as ht

"""Hierarchical representations for frame analysis: FrameWord --> FramePoles --> FrameAxes --> FrameSystem."""

class FrameWord:
    """Represents a single framing word. Mainly a wrapper."""

    def __init__(self, word, model=None):
        self.word = word
        self.model = model


class FramePole:
    """Represents a pole of the FrameAxis. Thus either positive or negative words."""

    def __init__(self, pole_name, words, model):
        self.pole_name = pole_name
        self.words = words
        self.initial_words = words  # For debugging
        self.model = model

    def compute(self):
        """Computes everything for its usage (e.g., centroid)."""
        self.retain_model_words_only()
        self.compute_centroid()
        return self

    def retain_model_words_only(self, log_removed=False):
        """Cleans the initial words to fit the supplied model."""
        pole_words = []
        for pole_word in self.words:
            if pole_word in self.model.key_to_index:
                pole_words.append(pole_word)
            else:
                if log_removed:
                    print(f"Word {pole_word} not in vocab")
        self.words = pole_words
        return pole_words

    def extract_vectors_from_model(self):
        """Extract the relevant vectors from the model. In same order as the words."""
        pole_vecs = []
        for pole_word in self.words:
            vec = self.model.get_vector(pole_word)
            pole_vecs.append(vec)
        print(len(pole_vecs))
        self.pole_vecs = pole_vecs
        return pole_vecs

    def compute_centroid(self):
        """Computes the centroid and vectors. Assumes valid vocabulary. Call `retain_model_words_only` beforehand."""
        pole_vecs = self.extract_vectors_from_model()
        centroid = np.mean(pole_vecs, axis=0)
        self.centroid = centroid
        return centroid


class FrameAxis:
    """Represents a Frame Axis, which is a Semantic Axis (SemAxis) with Bias and Intensity."""

    def __init__(self, name, pos_words, neg_words, wv_name, word_vectors):
        self.name = name
        self.pos_words = pos_words
        self.neg_words = neg_words
        self.wv_name = wv_name  # required to reproduce frame_axis
        self.word_vectors = word_vectors
        # TODO: compute axis
        self.axis = None
        self.baseline_bias = None
        self.model = word_vectors

        self.sim_cache = dict()  # use it to cache word similarities for reuse

    @classmethod
    def from_poles(cls, pos_pole, neg_pole):
        name = pos_pole.pole_name + "/" + neg_pole.pole_name
        assert pos_pole.model == neg_pole.model
        return cls(name, pos_pole.words, neg_pole.words, "", pos_pole.model)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pkl.load(f)

    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    def attach_model(self, model):
        self.model = model

    def detach_model(self):
        self.model = None

    def compute_word_sim(self, word):
        pass

    def compute(self):
        self.pos_words, _ = self.retain_words_in_model(self.pos_words, self.model)
        self.neg_words, _ = self.retain_words_in_model(self.neg_words, self.model)
        self.compute_axis()
        return self

    def compute_axis(self):
        pos_centroid, pos_vecs = self.compute_centroid(self.pos_words, self.word_vectors)
        neg_centroid, neg_vecs = self.compute_centroid(self.neg_words, self.word_vectors)
        self.pos_centroid = pos_centroid
        self.neg_centroid = neg_centroid
        axis = pos_centroid - neg_centroid
        self.axis = axis
        return axis

    def retain_words_in_model(self, initial_words, model=None):
        if not model:
            model = self.model

        words_in_vocab = []
        words_not_in_vocab = []
        for word in initial_words:
            if word in model.key_to_index:
                words_in_vocab.append(word)
            else:
                words_not_in_vocab.append(word)
        return words_in_vocab, words_not_in_vocab

    def compute_centroid(self, frame_words, model=None):
        """Assumes valid list of words."""
        if not model:
            model = self.model

        frame_vecs = []
        for frame_word in frame_words:
            assert frame_word in model.key_to_index
            vec = model.get_vector(frame_word)
            frame_vecs.append(vec)
        centroid = np.mean(frame_vecs, axis=0)
        return centroid, frame_vecs

    def compute_bias_document(self, doc, model=None):
        if not model:
            model = self.model

        word_vecs = []
        words = doc.split()
        for word in words:
            if not word in model.key_to_index:
                continue
            word_vec = model.get_vector(word)
            word_vecs.append(word_vec)
        if not word_vecs:
            return 0  # No bias when no words
        sims = model.cosine_similarities(self.axis, word_vecs)
        return np.sum(sims) / len(words)

    def compute_baseline_bias(self, docs, model=None):
        if not model:
            model = self.model

        doc_biases = []
        for doc in docs:
            doc_biases.append(self.compute_bias_document(doc, model))
        baseline_bias = np.mean(doc_biases)
        self.baseline_bias = baseline_bias
        return baseline_bias

    def compute_intensity_document(self, doc, baseline_bias=None):
        if not baseline_bias:
            baseline_bias = self.baseline_bias
        if not baseline_bias:
            raise ValueError("Neithher baseline_bias provided nor inherent to object.")
        word_vecs = []
        words = doc.split()
        for word in words:
            if not word in self.word_vectors.key_to_index:
                continue
            word_vec = self.word_vectors.get_vector(word)
            word_vecs.append(word_vec)
        if not word_vecs:
            return 0  # No bias when no words
        sims = self.word_vectors.cosine_similarities(self.axis, word_vecs)
        sim_dev = (sims - baseline_bias) ** 2
        return np.sum(sim_dev) / len(words)

    def compute_baseline_intensity(self, docs, baseline_bias=None, model=None):
        if not model:
            model = self.model

        doc_intensities = []
        for doc in docs:
            doc_intensity = self.compute_intensity_document(doc, baseline_bias)
            doc_intensities.append(doc_intensity)
        baseline_intensity = np.mean(doc_intensities)
        self.baseline_intensity = baseline_intensity
        return baseline_intensity

    def effect_size(self, corpus: pd.Series, num_bootstrap_samples=1000):
        corpus_bias = self.compute_baseline_bias(corpus)
        corpus_intensity = self.compute_baseline_intensity(corpus, baseline_bias=corpus_bias)
        boostrap_samples = [corpus.sample(n=len(corpus), replace=True) for _ in range(num_bootstrap_samples)]

        cum_sample_bias = 0
        cum_sample_intensity = 0
        for sample in tqdm.tqdm(boostrap_samples):
            sample = corpus.sample(n=len(corpus))
            bootstrapped_bias = self.compute_baseline_bias(sample)
            bootstrapped_intensity = self.compute_baseline_intensity(sample, baseline_bias=bootstrapped_bias)
            cum_sample_bias += bootstrapped_bias
            cum_sample_intensity += bootstrapped_intensity
        # Effect sizes for bias and intensity
        eta_bias = abs(corpus_bias - cum_sample_bias / num_bootstrap_samples)
        eta_intensity = abs(corpus_intensity - cum_sample_intensity / num_bootstrap_samples)

        EffectSize = namedtuple("EffectSize", ["eta_bias", "eta_intensity"])
        return EffectSize(eta_bias, eta_intensity)


def compute_bias(document, frame_vector, model):
    word_vecs = []
    words = document.split(" ")
    for word in words:
        if not word in model.key_to_index:
            continue
        word_vec = model.get_vector(word)
        word_vecs.append(word_vec)
    if not word_vecs:
        return 0  # No bias when no words
    sims = model.cosine_similarities(frame_vector, word_vecs)
    return np.sum(sims) / len(words)


def compute_intensity(document, frame_vector, model, baseline_bias):
    word_vecs = []
    words = document.split(" ")
    for word in words:
        if not word in model.key_to_index:
            continue
        word_vec = model.get_vector(word)
        word_vecs.append(word_vec)
    if not word_vecs:
        return 0  # No bias when no words
    sims = model.cosine_similarities(frame_vector, word_vecs)
    sim_dev = (sims - baseline_bias) ** 2
    return np.sum(sim_dev) / len(words)


class FrameSystem:
    """Represents a set of FrameAxes that form a complete system."""

    def __init__(self, frame_axes: Dict[str, FrameAxis]):
        self.frame_axes = frame_axes

    def transform_df(self, df: pd.DataFrame, text_col: str, model):
        for name, axis in self.frame_axes.items():
            pos_name, neg_name = name.split("/")
            axis_code = pos_name[:4]
            df[axis_code + "_bias"] = df[text_col].map(lambda x: compute_bias(x, axis.axis, model))
            baseline_bias = df[axis_code + "_bias"].mean()
            df[axis_code + "_inte"] = df[text_col].map(lambda x: compute_intensity(x, axis.axis, model, baseline_bias))
        return df

    def axes_ordered_by_effect_sizes(self, corpus: pd.Series, num_bootstrap_samples=1000, sort_key="eta_bias"):
        axes_effect_sizes = {}
        for name, axis in self.frame_axes.items():
            axes_effect_sizes[name] = axis.effect_size(corpus=corpus, num_bootstrap_samples=num_bootstrap_samples)
        return OrderedDict(sorted(axes_effect_sizes.items(), key=lambda x: -getattr(x[1], sort_key)))

    def compute_baseline_biases(self, df: pd.DataFrame, text_col: str, model):
        baseline_biases = {}
        for name, axis in self.frame_axes.items():
            pos_name, neg_name = name.split("/")  # TODO: move this informtion to FrameAxis
            axis_code = pos_name[:4]
            baseline_bias = df[text_col].map(lambda x: compute_bias(x, axis.axis, model)).mean()
            baseline_biases[name] = baseline_bias
        return baseline_biases

    def attach_model(self, model):
        self.model = model
        for name, axis in self.frame_axes.items():
            axis.attach_model(model)

    def compute(self):
        for name, axis in self.frame_axes.items():
            axis.compute()
        return self

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pkl.load(f)

    def save(self, filename):
        # Detach models before storing
        for _, frame_axis in self.frame_axes.items():
            frame_axis.detach_model()

        with open(filename, "wb") as f:
            pkl.dump(self, f)


# Helper classes

class FrameExperiment:
    """Combination of Dictionary Mapping, WordEmbeddings, and Dataset. Used to conduct evaluations."""

    def __init__(self, dictionary_mapping, wordembeddings, dataset):
        self.dictionary_mapping = dictionary_mapping
        self.wordembeddings = wordembeddings
        self.dataset = dataset


class FrameVisualization:
    """Used to visualize the frame system."""

    def __init__(self, frame_system: FrameSystem):
        self.frame_system = frame_system


def frame_axis_polarization(frame_axis):
    pos_centroid = frame_axis.pos_centroid
    neg_centroid = frame_axis.neg_centroid
    pos = frame_axis.pos_words
    neg = frame_axis.neg_words

    word_vectors = frame_axis.word_vectors

    inter_dist_pos = np.mean(word_vectors.distances(pos_centroid, pos))
    inter_dist_neg = np.mean(word_vectors.distances(neg_centroid, neg))
    between_dist_pos = np.mean(word_vectors.distances(pos_centroid, neg))
    between_dist_neg = np.mean(word_vectors.distances(neg_centroid, pos))
    print(f"{inter_dist_pos=} {inter_dist_neg=} {between_dist_pos=} {between_dist_neg=}")

    return (between_dist_pos + between_dist_neg) / 2 - (inter_dist_pos + inter_dist_neg) / 2

def print_hi(name):
    print(f'{Fore.GREEN}------------------------------- Start, {name} --------------------------------------------------{Style.RESET_ALL}')  # Press Strg+F8 to toggle the breakpoint.

def print_step(step):
    print(f'{Fore.GREEN}------------------------------- Next step is {step} ---------------------------------------------{Style.RESET_ALL}')

def print_inter_step(inter_step):
    print(f'{Fore.YELLOW}------------------------------- {inter_step} ---------------------------------------------{Style.RESET_ALL}')

def print_done():
    print(f'{Fore.GREEN}------------------------------- Script is finished ---------------------------------------------{Style.RESET_ALL}')

def visualize_frame_axis(pos_pole, neg_pole, oth_pole, model, reducer=None, add_kde=True, lim=4.2,
                         use_legend=False):
    if reducer is None:
        # Use reduced define outside
        reducer = embedding

    title = pos_pole.pole_name.capitalize() + "/" + neg_pole.pole_name.capitalize()
    pos_words = pos_pole.words
    neg_words = neg_pole.words
    oth_words = oth_pole.words

    pos_centroid = pos_pole.compute_centroid()
    pos_vecs = pos_pole.pole_vecs
    neg_centroid = neg_pole.centroid
    neg_vecs = neg_pole.pole_vecs
    oth_centroid = oth_pole.centroid
    oth_vecs = oth_pole.pole_vecs

    projection = reducer.transform(pos_vecs + neg_vecs + oth_vecs + [pos_centroid] + [neg_centroid])
    print(projection.shape)

    labels = ["virtues"] * len(pos_vecs) + ["vices"] * len(neg_vecs) + ["other"] * len(oth_vecs) + [
        "pos_centroid"] + ["neg_centroid"]

    x1, y1 = projection[-2]
    x2, y2 = projection[-1]

    dx, dy = x1 - x2, y1 - y2

    x = projection[:-2, 0]
    y = projection[:-2, 1]
    labels = labels[:-2]  # ignore centroids

    plot_df = pd.DataFrame(projection[:-2, :], columns=["x", "y"])
    plot_df = pd.concat([plot_df, pd.Series(labels, name="labels")], axis=1)

    ax = sns.scatterplot(data=plot_df, x="x", y="y", hue="labels", style="labels", palette="colorblind",
                         legend=use_legend, size=[1] * len(x))
    if add_kde:
        sns.kdeplot(data=plot_df[plot_df["labels"] != "other"], x="x", y="y", hue="labels", legend=use_legend,
                    levels=3, thresh=.33)
    if use_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:-1], labels=labels[:-1])

    ax = plt.gca()
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
    plt.quiver(x2, y2, dx, dy, angles='xy', scale_units='xy', scale=1)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    sns.despine()


if __name__ == '__main__':
    print_hi('vallidate with umap')
    # https://derpylz.github.io/semantle_umap/

    print_step("load HanoverTagger for lammaziation")
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')

    print_step("load model")

    load_folder = "model/random/final/"
    save_folder = "pic/vallidation/"

    gender_word = [
                    'Tochter', 'Mutter', 'Frau', 'weiblich', 'Schwester', 'mama', 'Tante', 'female_name',
                    'Sohn', 'Vater', 'Mann', 'männlich', 'Bruder', 'Papa', 'Onkel', 'male_name'
                  ]
    female_word = ['Tochter', 'Mutter', 'Frau', 'weiblich', 'Schwester', 'mama', 'Tante', 'female_name']
    male_word = ['Sohn', 'Vater', 'Mann', 'männlich', 'Bruder', 'Papa', 'Onkel', 'male_name']


    text_model = models.Word2Vec.load(f"{load_folder}model")
    print(text_model)

    gender_identifier = []

    for wrd in text_model.wv.index_to_key:
        if wrd in female_word:
            gender_identifier.append("female")
        elif wrd in male_word:
            gender_identifier.append("male")
        else:
            gender_identifier.append("word")

    print_step("Calculate Frame Pole")
    all_words = text_model.wv.index_to_key
    fp_all = FramePole("all", all_words, text_model.wv)
    fp_all.compute_centroid()

    print_step("Reduce to two dimations")
    fitted_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42).fit(fp_all.pole_vecs)
    print_step("dimations are reduced")

    projection = fitted_reducer.transform(fp_all.pole_vecs)
    x = projection[:, 0]
    y = projection[:, 1]

    plot_df = pd.DataFrame(projection, columns=["x", "y"])
    plot_df['Gender'] = gender_identifier
    plot_df['Token'] = text_model.wv.index_to_key

    print(plot_df)

    dict_color = {'female': "green", 'male': "red", 'word': "grey"}

    options = ['female', 'male']
    plot_df_only_gender = plot_df[plot_df['Gender'].isin(options)]
    print(plot_df_only_gender)

    plt.figure(figsize=(5, 5))

    sns.kdeplot(data=plot_df_only_gender,
                x="x", y="y",
                hue="Gender",
                levels=4,
                #thresh=0.3,
                palette=dict_color,
                fill=True, alpha=.1
                )  # , palette=sns.color_palette(), fill=True)#, legend=False)#,? hue=)#, )

    ax = sns.scatterplot(data=plot_df, x="x", y="y",
                         hue="Gender", hue_order=['female', 'male', 'word'],
                         palette=dict_color,
                         style="Gender", style_order=['female', 'male', 'word'],
                         size="Gender", sizes={'female': 20, 'male': 20, 'word': 1},
                         markers={'female': "P", 'male': "X", 'word': "o"},
                         alpha=1
                         )
    plt.title(f"Distribution word vectors")
    plt.tight_layout()
    plt.savefig(f"{save_folder}visual_validation_without_labels.png")

    for i, txt in enumerate(plot_df['Token']):
        print(f"{i} = {txt}, x={plot_df['x'][i]}, y={plot_df['y'][i]}")
        if txt in gender_word:
           ax.annotate(txt, (plot_df['x'][i], plot_df['y'][i]))

    plt.title(f"Distribution word vectors")
    plt.tight_layout()
    plt.savefig(f"{save_folder}visual_validation.png")

    print_done()
