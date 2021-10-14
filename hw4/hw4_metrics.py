from scipy import sparse
import numpy as np

corpus_count_a = sparse.load_npz('sparse_matrix_countvect_a.npz')
corpus_count_q = sparse.load_npz('sparse_matrix_countvect_q.npz')
corpus_tfidf_a = sparse.load_npz('sparse_matrix_tfidfvect_a.npz')
corpus_tfidf_q = sparse.load_npz('sparse_matrix_tfidfvect_q.npz')
corpus_bm25_a = sparse.load_npz('sparse_matrix_bm25_a.npz')
corpus_bm25_q = sparse.load_npz('sparse_matrix_bm25_q.npz')
corpus_w2v_a = sparse.load_npz('sparse_matrix_w2v_a.npz')
corpus_w2v_q = sparse.load_npz('sparse_matrix_w2v_q.npz')
corpus_bert_a = sparse.load_npz('sparse_matrix_bert_a.npz')
corpus_bert_q = sparse.load_npz('sparse_matrix_bert_q.npz')


def get_metric_score(corpus_of_answers, corpus_of_questions):
    matrix = np.dot(corpus_of_answers, corpus_of_questions.T)
    score = 0
    sorted_matrix = np.argsort(-1 * matrix.toarray(), axis=1)
    for index, row in enumerate(sorted_matrix):
        if index in row[:5]:
            score += 1
    metric_score = score / len(sorted_matrix)
    print('number of right answers:', score)
    print('average score of metric:', round(metric_score, 3))


print('CountVectorizer')
get_metric_score(corpus_count_a, corpus_count_q)
print('TfidfVectorizer')
get_metric_score(corpus_tfidf_a, corpus_tfidf_q)
print('BM25')
get_metric_score(corpus_bm25_a, corpus_bm25_q)
print('Fasttext')
get_metric_score(corpus_w2v_a, corpus_w2v_q)
print('Bert')
get_metric_score(corpus_bert_a, corpus_bert_q)
