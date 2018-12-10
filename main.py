import glob

import chardet
import luigi
import luigi.format
import pickle
import subprocess
import pandas as pd
import numpy as np
import os
from model.scdv import SCDV
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

from sklearn.metrics import classification_report, accuracy_score, make_scorer


def make_target(file_path):
    extension = os.path.splitext(file_path)[1]
    if extension == '.pkl':
        return luigi.LocalTarget(file_path, format=luigi.format.Nop)
    return luigi.LocalTarget(file_path)


def dump(target: luigi.LocalTarget, obj):
    extension = os.path.splitext(target.path)[1]
    with target.open('w') as f:
        if extension == '.pkl':
            f.write(pickle.dumps(obj, protocol=4))
        else:
            f.write(str(obj))


def load(target: luigi.LocalTarget):
    with target.open('r') as f:
        return pickle.load(f)


def tokenize(file_path):
    p = subprocess.run(['mecab', '-Owakati', file_path],
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell=False)
    try:
        lines = p.stdout.decode(chardet.detect(p.stdout)["encoding"])
        return lines.split()
    except:
        return None


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


if __name__ == '__main__':
    luigi.run()
