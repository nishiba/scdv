from typing import Dict, Any, List

import itertools
import numpy as np
import sklearn
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, TfidfModel
from sklearn.mixture import GaussianMixture


class SCDV(object):
    """ This is a model which is described in "SCDV : Sparse Composite Document Vectors using soft clustering over distributional representations"
    See https://arxiv.org/pdf/1612.06778.pdf for details
    
    """

    def __init__(self, documents: List[List[str]], embedding_size: int, cluster_size: int, sparsity_percentage: float,
                 word2vec_parameters: Dict[Any, Any], gaussian_mixture_parameters: Dict[Any, Any],
                 dictionary_filter_parameters: Dict[Any, Any]) -> None:
        """
        
        :param documents: documents for training.
        :param embedding_size: word embedding size.
        :param cluster_size:  word cluster size.
        :param sparsity_percentage: sparsity percentage. This must be in [0, 1].
        :param word2vec_parameters: parameters to build `gensim.models.Word2Vec`. Please see `gensim.models.Word2Vec.__init__` for details.
        :param gaussian_mixture_parameters: parameters to build `sklearn.mixture.GaussianMixture`. Please see `sklearn.mixture.GaussianMixture.__init__` for details.
        :param dictionary_filter_parameters: parameters for `gensim.corpora.Dictionary.filter_extremes`. Please see `gensim.corpora.Dictionary.filter_extremes` for details.
        """
        self._dictionary = self._build_dictionary(documents, dictionary_filter_parameters)
        vocabulary_size = len(self._dictionary.token2id)

        self._word_embeddings = self._build_word_embeddings(documents, self._dictionary, embedding_size,
                                                            word2vec_parameters)
        assert self._word_embeddings.shape == (vocabulary_size, embedding_size)

        self._word_cluster_probabilities = self._build_word_cluster_probabilities(self._word_embeddings, cluster_size,
                                                                                  gaussian_mixture_parameters)
        assert self._word_cluster_probabilities.shape == (vocabulary_size, cluster_size)

        self._idf = self._build_idf(documents, self._dictionary)
        assert self._idf.shape == (vocabulary_size, )

        word_cluster_vectors = self._build_word_cluster_vectors(self._word_embeddings, self._word_cluster_probabilities)
        assert word_cluster_vectors.shape == (vocabulary_size, cluster_size, embedding_size)

        word_topic_vectors = self._build_word_topic_vectors(self._idf, word_cluster_vectors)
        assert word_topic_vectors.shape == (vocabulary_size, (cluster_size * embedding_size))

        document_vectors = self._build_document_vectors(word_topic_vectors, self._dictionary, documents)
        assert document_vectors.shape == (len(documents), cluster_size * embedding_size)

        self._sparse_threshold = self._build_sparsity_threshold(document_vectors, sparsity_percentage)

    @property
    def dictionary(self) -> Dictionary:
        return self._dictionary

    @property
    def word_cluster_probabilities(self) -> np.ndarray:
        return self._word_cluster_probabilities

    def infer_vector(self, new_documents: List[List[str]], l2_normalize: bool = True) -> np.ndarray:
        word_cluster_vectors = self._build_word_cluster_vectors(self._word_embeddings, self._word_cluster_probabilities)
        word_topic_vectors = self._build_word_topic_vectors(self._idf, word_cluster_vectors)
        document_vectors = self._build_document_vectors(word_topic_vectors, self._dictionary, new_documents)
        return self._build_scdv_vectors(document_vectors, self._sparse_threshold, l2_normalize)

    @staticmethod
    def _build_dictionary(documents: List[List[str]], filter_parameters: Dict[Any, Any]) -> Dictionary:
        d = Dictionary(documents)
        d.filter_extremes(**filter_parameters)
        return d

    @staticmethod
    def _build_word_embeddings(documents: List[List[str]], dictionary: Dictionary, embedding_size: int,
                               word2vec_parameters: Dict[Any, Any]) -> np.ndarray:
        w2v = Word2Vec(documents, size=embedding_size, **word2vec_parameters)
        embeddings = np.zeros((len(dictionary.token2id), w2v.vector_size))
        for token, idx in dictionary.token2id.items():
            embeddings[idx] = w2v.wv[token]
        return sklearn.preprocessing.normalize(embeddings, axis=1, norm='l2')

    @staticmethod
    def _build_word_cluster_probabilities(word_embeddings: np.ndarray, cluster_size: int,
                                          gaussian_mixture_parameters: Dict[Any, Any]) -> np.ndarray:
        gm = GaussianMixture(n_components=cluster_size, **gaussian_mixture_parameters)
        gm.fit(word_embeddings)
        return gm.predict_proba(word_embeddings)

    @staticmethod
    def _build_idf(documents: List[List[str]], dictionary: Dictionary) -> np.ndarray:
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        model = TfidfModel(corpus=corpus, dictionary=dictionary)
        idf = np.zeros(len(dictionary.token2id))
        for idx, value in model.idfs.items():
            idf[idx] = value
        return idf

    @staticmethod
    def _build_word_cluster_vectors(word_embeddings: np.ndarray, word_cluster_probabilities: np.ndarray) -> np.ndarray:
        vocabulary_size, embedding_size = word_embeddings.shape
        cluster_size = word_cluster_probabilities.shape[1]
        assert vocabulary_size == word_cluster_probabilities.shape[0]

        wcv = np.zeros((vocabulary_size, cluster_size, embedding_size))
        wcp = word_cluster_probabilities
        for v, c in itertools.product(range(vocabulary_size), range(cluster_size)):
            wcv[v][c] = wcp[v][c] * word_embeddings[v]
        return wcv

    @staticmethod
    def _build_word_topic_vectors(idf: np.ndarray, word_cluster_vectors: np.ndarray) -> np.ndarray:
        vocabulary_size, cluster_size, embedding_size = word_cluster_vectors.shape
        assert vocabulary_size == idf.shape[0]

        wtv = np.zeros((vocabulary_size, cluster_size * embedding_size))
        for v in range(vocabulary_size):
            wtv[v] = idf[v] * word_cluster_vectors[v].flatten()
        return wtv

    @staticmethod
    def _build_document_vectors(word_topic_vectors: np.ndarray, dictionary: Dictionary,
                                documents: List[List[str]]) -> np.ndarray:
        return np.array([
            np.sum([word_topic_vectors[idx] * count for idx, count in dictionary.doc2bow(d)], axis=0) for d in documents
        ])

    @staticmethod
    def _build_sparsity_threshold(document_vectors: np.ndarray, sparsity_percentage) -> float:
        def _abs_average_max(m: np.ndarray) -> float:
            return np.abs(np.average(np.max(m, axis=1)))

        t = 0.5 * (_abs_average_max(document_vectors) + _abs_average_max(-document_vectors))
        return sparsity_percentage * t

    @staticmethod
    def _build_scdv_vectors(document_vectors: np.ndarray, sparsity_threshold: float, l2_normalize: bool) -> np.ndarray:
        close_to_zero = np.abs(document_vectors) < sparsity_threshold
        document_vectors[close_to_zero] = 0.0
        if not l2_normalize:
            return document_vectors

        return sklearn.preprocessing.normalize(document_vectors, axis=1, norm='l2')
