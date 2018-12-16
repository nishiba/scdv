import glob
import operator
from typing import Dict, List

import itertools
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from gensim.models import FastText, Word2Vec

from model.scdv import SCDV
from task.util import *


class PrepareLivedoorNewsData(luigi.Task):
    def output(self):
        return make_target('output/livedoor_news_data.pkl')

    def run(self):
        categories = [
            'dokujo-tsushin', 'it-life-hack', 'kaden-channel', 'livedoor-homme', 'movie-enter', 'peachy', 'smax',
            'sports-watch', 'topic-news'
        ]

        data = pd.DataFrame([(c, tokenize(path)) for c in categories for path in glob.glob(f'data/text/{c}/*.txt')],
                            columns=['category', 'text'])
        data.dropna(inplace=True)
        dump(self.output(), data)


class TrainFastText(luigi.Task):
    def requires(self):
        return PrepareLivedoorNewsData()

    def output(self):
        return make_target('output/fasttext.pkl')

    def run(self):
        data = load(self.input())  # type: pd.DataFrame
        data = data.sample(frac=1).reset_index(drop=True)

        documents = data['text'].tolist()
        model = FastText(sentences=documents, size=200)
        dump(self.output(), model)


class TrainWord2Vec(luigi.Task):
    def requires(self):
        return PrepareLivedoorNewsData()

    def output(self):
        return make_target('output/word2vec.pkl')

    def run(self):
        data = load(self.input())  # type: pd.DataFrame
        data = data.sample(frac=1).reset_index(drop=True)

        documents = data['text'].tolist()
        model = Word2Vec(documents, size=200)
        dump(self.output(), model)


class TrainSCDV(luigi.Task):
    def requires(self):
        return PrepareLivedoorNewsData()

    def output(self):
        return make_target('output/scdv.pkl')

    def run(self):
        data = load(self.input())  # type: pd.DataFrame
        data = data.sample(frac=1).reset_index(drop=True)

        documents = data['text'].tolist()
        embedding_size = 200
        cluster_size = 60
        sparsity_percentage = 0.04
        word2vec_parameters = dict()
        gaussian_mixture_parameters = dict()
        dictionary_filter_parameters = dict()

        model = SCDV(
            documents=documents,
            embedding_size=embedding_size,
            cluster_size=cluster_size,
            sparsity_percentage=sparsity_percentage,
            word2vec_parameters=word2vec_parameters,
            gaussian_mixture_parameters=gaussian_mixture_parameters,
            dictionary_filter_parameters=dictionary_filter_parameters)
        dump(self.output(), model)


class PrepareClassificationData(luigi.Task):
    def requires(self):
        return dict(data=PrepareLivedoorNewsData(), model=TrainSCDV())

    def output(self):
        return make_target('output/classification_data.pkl')

    def run(self):
        data = load(self.input()['data'])  # type: pd.DataFrame
        model = load(self.input()['model'])  # type: SCDV

        data['embedding'] = list(model.infer_vector(data['text'].tolist(), l2_normalize=True))
        data = data[['category', 'embedding']].copy()
        data['category'] = data['category'].astype('category')
        data['category_code'] = data['category'].cat.codes

        dump(self.output(), data)


class PrepareSimilarityData(luigi.Task):
    def requires(self):
        return dict(data=PrepareLivedoorNewsData(), model=TrainSCDV())

    def output(self):
        return dict(
            examples=make_target('output/similarity_data.pkl'),
            index2embedding=make_target('output/index2embedding.pkl'))

    def run(self):
        data = load(self.input()['data'])  # type: pd.DataFrame
        model = load(self.input()['model'])  # type: SCDV
        data.reset_index(inplace=True)
        data['category'] = data['category'].astype('category')
        data['category_code'] = data['category'].cat.codes
        category_codes = list(set(data['category_code'].tolist()))
        category2indices = dict(
            zip(category_codes, [data[data['category_code'] == c]['index'].tolist() for c in category_codes]))
        positive_examples = self._positive_examples(category2indices, n_samples=data.shape[0])
        negative_examples = self._negative_examples(category2indices, n_samples=data.shape[0])
        examples = pd.DataFrame(positive_examples + negative_examples, columns=['index0', 'index1', 'similarity'])

        index2embedding = dict(
            zip(data['index'].tolist(), list(model.infer_vector(data['text'].tolist(), l2_normalize=True))))

        dump(self.output()['examples'], examples)
        dump(self.output()['index2embedding'], index2embedding)

    @staticmethod
    def _positive_examples(category2indices: Dict[int, List[int]], n_samples: int):
        def _make(category: int):
            targets = category2indices[category]
            return zip(np.random.choice(targets, n_samples), np.random.choice(targets, n_samples), np.ones(n_samples))

        return list(itertools.chain.from_iterable([_make(category) for category in category2indices.keys()]))

    @staticmethod
    def _negative_examples(category2indices: Dict[int, List[int]], n_samples: int):
        def _make_positive(category: int):
            return np.random.choice(category2indices[category], n_samples)

        def _make_negative(category: int):
            negative_categories = set(category2indices.keys()) - {category}
            indices = list(itertools.chain.from_iterable([category2indices[c] for c in negative_categories]))
            return np.random.choice(indices, n_samples)

        return list(
            itertools.chain.from_iterable(
                [zip(_make_positive(c), _make_negative(c), [0] * n_samples) for c in category2indices.keys()]))


class TrainClassificationModel(luigi.Task):
    def requires(self):
        return PrepareClassificationData()

    def output(self):
        return dict(
            scores=make_target('output/classification_scores.pkl'),
            model=make_target('output/classification_model.pkl'))

    def run(self):
        data = load(self.input())  # type: pd.DataFrame
        data = data.sample(frac=1).reset_index(drop=True)

        x = data['embedding'].tolist()
        y = data['category_code'].tolist()
        model = lgb.LGBMClassifier(objective="multiclass")

        scores = []

        def _scoring(y_true, y_pred):
            scores.append(classification_report(y_true, y_pred))
            return accuracy_score(y_true, y_pred)

        cross_val_score(model, x, y, cv=3, scoring=make_scorer(_scoring))
        dump(self.output()['scores'], scores)
        model.fit(x, y)
        dump(self.output()['model'], model)


class TrainSimilarityModel(luigi.Task):
    def requires(self):
        return PrepareSimilarityData()

    def output(self):
        return dict(
            scores=make_target('output/similarity_scores.pkl'), model=make_target('output/similarity_model.pkl'))

    def run(self):
        data = load(self.input()['examples'])  # type: pd.DataFrame
        index2embedding = load(self.input()['index2embedding'])  # type: pd.DataFrame
        data = data.sample(frac=1).reset_index(drop=True)

        x = np.array([
            np.multiply(index2embedding[i0], index2embedding[i1])
            for i0, i1 in zip(data['index0'].tolist(), data['index1'].tolist())
        ])

        y = data['similarity'].tolist()
        model = lgb.LGBMClassifier(objective="binary")

        scores = []

        def _scoring(y_true, y_pred):
            scores.append(classification_report(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            return accuracy_score(y_true, y_pred)

        cross_val_score(model, x, y, cv=3, scoring=make_scorer(_scoring))
        dump(self.output()['scores'], scores)
        model.fit(x, y)
        dump(self.output()['model'], model)


class ReportClassificationResults(luigi.Task):
    def requires(self):
        return TrainClassificationModel()

    def output(self):
        return make_target('output/results.txt')

    def run(self):
        score_texts = load(self.input()['scores'])
        scores = np.array([self._extract_average(text) for text in score_texts])
        averages = dict(zip(['precision', 'recall', 'f1-score', 'support'], np.average(scores, axis=0)))
        dump(self.output(), averages)

    @staticmethod
    def _extract_average(score_text: str):
        # return 'precision', 'recall', 'f1-score', 'support'
        return [float(x) for x in score_text.split()[-4:]]


class AnalyseFeatureImportance(luigi.Task):
    def requires(self):
        return dict(clf=TrainClassificationModel(), scdv=TrainSCDV())

    def output(self):
        return make_target('output/analyze_feature_importance.txt')

    def run(self):
        importance = self._importance()
        top_words = self._top_words()
        data = pd.merge(importance, top_words, on='cluster')
        data.sort_values(by='importance', ascending=False, inplace=True)
        dump(self.output(), data.to_csv(sep='|', index=False))

    def _top_words(self) -> pd.DataFrame:
        model = load(self.input()['scdv'])  # type: SCDV
        dictionary = model.dictionary
        prob = pd.DataFrame(model.word_cluster_probabilities)
        prob['word'] = list(zip(*sorted(dictionary.token2id.items(), key=operator.itemgetter(1))))[0]
        cluster_size = 60

        return pd.DataFrame(
            [(c, prob.sort_values(by=c, ascending=False)['word'].tolist()[:20]) for c in range(cluster_size)],
            columns=['cluster', 'word'])

    def _importance(self) -> pd.DataFrame:
        model = load(self.input()['clf']['model'])
        cluster_size = 60
        element_importance = model.feature_importances_
        cluster_importance = list(
            zip(range(cluster_size), [np.sum(x) for x in np.array_split(element_importance, cluster_size)]))
        return pd.DataFrame(cluster_importance, columns=['cluster', 'importance'])
