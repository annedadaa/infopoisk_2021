import re
import nltk
import json
import pymorphy2
import numpy as np
from scipy import sparse
from pymystem3 import Mystem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')
stopword = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
m = Mystem()
tf_vectorizer = TfidfVectorizer(use_idf=False)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
count_vectorizer = CountVectorizer()


def get_corpus(filename):
    corpus = []
    with open(filename, 'r') as f:
        raw_corpus = list(f)[:50000]
    for item in range(0, len(raw_corpus)):
        data = json.loads(raw_corpus[item])
        largest = 0
        index = None
        if len(data['answers']) > 0:
            for num, answer in enumerate(data['answers']):
                rating = answer['author_rating']['value']
                if len(rating) > 0 and largest < int(rating):
                    largest = int(rating)
                    index = num
            answer_text = (data['answers'][index]['text'])
            if len(answer_text) > 0:
                corpus.append(answer_text)
        else:
            pass
    return corpus


def preprocessing(any_string):
    any_text = str(any_string)
    tokens = m.lemmatize(any_text.lower())
    tokens = [token for token in tokens if token != " "
              and len(token) >= 3
              and token.isdigit() is False]
    any_text = " ".join(tokens)
    any_text = re.sub("[^а-яА-Я]+", ' ', any_text)
    return any_text


def bm25(corpus):
    x_count_vec = count_vectorizer.fit_transform(corpus)
    x_tf_vec = tf_vectorizer.fit_transform(corpus)
    tfidf_vectorizer.fit_transform(corpus)
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
    return docs_matrix


def most_relevant(query, docs_matrix, corpus):
    prep_query = preprocessing(query)
    query_count_vec = count_vectorizer.transform([prep_query])
    similarity = np.dot(docs_matrix, query_count_vec.T)
    sorted_scores_index = np.argsort(similarity.toarray(), axis=0)[::-1]
    corpus = np.array(corpus)
    results = corpus[sorted_scores_index.ravel()]
    return results


if __name__ == '__main__':
    final_filename = 'questions_about_love.jsonl'
    final_corpus = get_corpus(final_filename)
    prep_corpus = []
    for text in final_corpus:
        text = preprocessing(text)
        prep_corpus.append(text)
    final_docs_matrix = bm25(prep_corpus)
    while True:
        final_query = input("введите запрос: ")
        final_results = most_relevant(final_query, final_docs_matrix, final_corpus)
        print('топ-10 релевантных документов:')
        for res in final_results[0:10]:
            print(res)
