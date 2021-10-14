import re
import nltk
import json
import torch
import pickle
import pymorphy2
import numpy as np
from scipy import sparse
from pymystem3 import Mystem
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')

stopword = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
m = Mystem()
tf_vectorizer = TfidfVectorizer(use_idf=False)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
count_vectorizer_bm25 = CountVectorizer()
count_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
fasttext_model_file = 'araneum_none_fasttextcbow_300_5_2018.model'
fasttext_model = KeyedVectors.load(fasttext_model_file)
rubert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
rubert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

corpus_count = sparse.load_npz('sparse_matrix_countvect_a.npz')
count_vectorizer_final = pickle.load(open("count_vectorizer.pickle", "rb"))
corpus_tfidf = sparse.load_npz('sparse_matrix_tfidfvect_a.npz')
tfidf_vectorizer_final = pickle.load(open("tfidf_vectorizer.pickle", "rb"))
corpus_bm25 = sparse.load_npz('sparse_matrix_bm25_a.npz')
bm25_vectorizer_final = pickle.load(open("bm25_vectorizer.pickle", "rb"))
corpus_w2v = sparse.load_npz('sparse_matrix_w2v_a.npz')
corpus_bert = sparse.load_npz('sparse_matrix_bert_a.npz')


def get_corpus(filename, number):
    answer_corpus = []
    question_corpus = []
    with open(filename, 'r') as f:
        raw_corpus = list(f)[:number]
    for item in range(0, len(raw_corpus)):
        data = json.loads(raw_corpus[item])
        largest = 0
        index = None
        if len(data['answers']) > 0:
            question = data['question'] + ' ' + data['comment']
            question_corpus.append(question)
            for num, answer in enumerate(data['answers']):
                rating = answer['author_rating']['value']
                if len(rating) > 0 and largest < int(rating):
                    largest = int(rating)
                    index = num
            answer_text = (data['answers'][index]['text'])
            if len(answer_text) > 0:
                answer_corpus.append(answer_text)
        else:
            pass
    return answer_corpus, question_corpus


def preprocessing(corpus):
    prep_corpus = []
    for text in corpus:
        any_text = str(text)
        tokens = m.lemmatize(any_text.lower())
        tokens = [token for token in tokens if token != " "
                    and len(token) >= 3
                    and token.isdigit() is False]
        any_text = " ".join(tokens)
        any_text = re.sub("[^а-яА-Я]+", ' ', any_text)
        prep_corpus.append(any_text)
    return prep_corpus


def countvect(prep_corpus):
    docs_matrix = count_vectorizer.fit_transform(prep_corpus)
    return docs_matrix  # выводится спарс-матрица


def tfidfvect(prep_corpus):
    docs_matrix = tfidf_vectorizer.fit_transform(prep_corpus)
    return docs_matrix  # выводится спарс-матрица


def bm25(prep_corpus):
    x_count_vec = count_vectorizer_bm25.fit_transform(prep_corpus)
    x_tf_vec = tf_vectorizer.fit_transform(prep_corpus)
    tfidf_vectorizer.fit_transform(prep_corpus)
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vec
    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = x_count_vec.sum(axis=1).mean()
    b_1 = (k * (1 - b + b * len_d / avdl))
    b_1 = np.expand_dims(b_1, axis=-1)
    values, rows, cols = [], [], []
    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        x = idf[0][j] * tf[i, j] * (k + 1)
        y = tf[i, j] + b_1[i]
        bm25_values = x / y
        values.append(bm25_values[0][0])
    docs_matrix = sparse.csr_matrix((values, (rows, cols)))
    return docs_matrix  # выводится спарс-матрица


def fasttext(prep_corpus):
    vectors = []
    for text in prep_corpus:
        tokens = text.split()
        tokens_vectors = np.zeros((len(tokens), fasttext_model.vector_size))
        for id, token in enumerate(tokens):
            tokens_vectors[id] = fasttext_model[token]
        if tokens_vectors.shape[0] != 0:
            vector = np.mean(tokens_vectors, axis=0)
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    docs_matrix = sparse.csr_matrix(vectors)
    return docs_matrix  # выводится спарс-матрица


def embed_bert_cls(corpus, model=rubert_model, tokenizer=rubert_tokenizer):
    vectors = []
    for text in corpus:
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        emb_array = embeddings[0].cpu().numpy()
        vectors.append(emb_array)
    docs_matrix = sparse.csr_matrix(vectors)
    return docs_matrix  # выводится спарс-матрица


def get_similarity(corpus, query, raw_corpus):
    similarity = np.dot(corpus, query.T)
    sorted_scores_index = np.argsort(similarity.toarray(), axis=0)[::-1]
    raw_corpus = np.array(raw_corpus)
    results = raw_corpus[sorted_scores_index.ravel()]
    return results


if __name__ == '__main__':
    final_corpus = get_corpus('questions_about_love.jsonl', 10000)[0]
    while True:
        final_query = input("введите запрос: ")
        final_model = input("введите число от 1 до 5 (соотв. Count, Tfidf, BM25, W2V, Bert) : ")
        if final_model == '1':
            prep_query = preprocessing([final_query])
            query_vec = count_vectorizer_final.transform(prep_query)
            results = get_similarity(corpus_count, query_vec, final_corpus)
            print('топ-10 релевантных документов:')
            for res in results[0:10]:
                print(res)
        if final_model == '2':
            prep_query = preprocessing([final_query])
            query_vec = tfidf_vectorizer_final.transform(prep_query)
            results = get_similarity(corpus_tfidf, query_vec, final_corpus)
            print('топ-10 релевантных документов:')
            for res in results[0:10]:
                print(res)
        if final_model == '3':
            prep_query = preprocessing([final_query])
            prep_query = "".join(prep_query)
            query_vec = bm25_vectorizer_final.transform([prep_query])
            results = get_similarity(corpus_bm25, query_vec, final_corpus)
            print('топ-10 релевантных документов:')
            for res in results[0:10]:
                print(res)
        if final_model == '4':
            prep_query = preprocessing([final_query])
            query_vec = fasttext(prep_query)
            results = get_similarity(corpus_w2v, query_vec, final_corpus)
            print('топ-10 релевантных документов:')
            for res in results[0:10]:
                print(res)
        if final_model == '5':
            query_vec = embed_bert_cls([final_query], model=rubert_model, tokenizer=rubert_tokenizer)
            results = get_similarity(corpus_bert, query_vec, final_corpus)
            print('топ-10 релевантных документов:')
            for res in results[0:10]:
                print(res)



