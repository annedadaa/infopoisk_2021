import re
import torch
import pickle
import streamlit as st
import numpy as np
import time
from scipy import sparse
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
import pymorphy2


@st.cache(allow_output_mutation=True)
def get_corpus():
    with open("answer_file.txt", 'r', encoding='utf-8') as f:
        love_corpus = [line.rstrip("\n") for line in f]
    count_corpus = sparse.load_npz('sparse_matrix_countvect_q.npz')
    tfidf_corpus = sparse.load_npz('sparse_matrix_tfidfvect_q.npz')
    bm25_corpus = sparse.load_npz('bm25_matrix_new.npz')
    fasttext_corpus = sparse.load_npz('sparse_matrix_w2v_q.npz')
    bert_corpus = sparse.load_npz('sparse_matrix_bert_q.npz')
    morph = pymorphy2.MorphAnalyzer()
    return love_corpus, count_corpus, tfidf_corpus, bm25_corpus, fasttext_corpus, bert_corpus, morph


@st.cache(allow_output_mutation=True)
def get_models():
    count_v = pickle.load(open("count_vectorizer.pickle", "rb"))
    tfidf_v = pickle.load(open("tfidf_vectorizer.pickle", "rb"))
    bm25_v = pickle.load(open("bm25_vectorizer_new.pickle", "rb"))
    fasttext_model_file = 'araneum_none_fasttextcbow_300_5_2018.model'
    fasttext_m = KeyedVectors.load(fasttext_model_file)
    bert_t = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny", TOKENIZERS_PARALLELISM=True)
    bert_m = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    return count_v, tfidf_v, bm25_v, fasttext_m, bert_t, bert_m


love_corpus, count_corpus, tfidf_corpus, bm25_corpus, fasttext_corpus, bert_corpus, morph = get_corpus()
count_v, tfidf_v, bm25_v, fasttext_m, bert_t, bert_m = get_models()

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://photogora.ru/img/product/thumb/14504/602fca1f93ac68021106479115502196.jpg")
    }
   .sidebar .sidebar-content {
        background: url("https://photogora.ru/img/product/thumb/14504/602fca1f93ac68021106479115502196.jpg")
    </style>
    """,
    unsafe_allow_html=True
)
st.title('Все о любви от мейл.ру 💌')


def preprocessing(corpus):
    prep_corpus = []
    for text in corpus:
        any_text = str(text.lower())
        any_text = any_text.split(' ')  # for pymorphy
        #tokens = m.lemmatize(any_text.lower())
        tokens = [morph.parse(word)[0].normal_form for word in any_text]
        tokens = [token for token in tokens if token != " "
                    and len(token) >= 3
                    and token.isdigit() is False]
        any_text = " ".join(tokens)
        any_text = re.sub("[^а-яА-Я]+", ' ', any_text)
        prep_corpus.append(any_text)
    return prep_corpus


def fasttext(prep_corpus, fasttext_model):
    vectors = []
    for text in prep_corpus:
        tokens = text.split()
        tokens_vectors = np.zeros((len(tokens), fasttext_model.vector_size))
        vector = np.zeros((fasttext_model.vector_size,))
        for idx, token in enumerate(tokens):
            tokens_vectors[idx] = fasttext_model[token]
        if tokens_vectors.shape[0] != 0:
            vector = np.mean(tokens_vectors, axis=0)
            vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    docs_matrix = sparse.csr_matrix(vectors)
    return docs_matrix  # выводится спарс-матрица


def embed_bert_cls(corpus, model, tokenizer):
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
    similarity = np.dot(corpus, query.T).toarray()
    sorted_scores_index = np.argsort(similarity, axis=0)[::-1]
    raw_corpus = np.array(raw_corpus)
    results = raw_corpus[sorted_scores_index.ravel()]
    return results


def search_answers(model, query):
    if model == 'CountVectorizer':
        prep_query = preprocessing([query])
        query_vec = count_v.transform(prep_query)
        results = get_similarity(count_corpus, query_vec, love_corpus)
        return results
    if model == 'TfidfVectorizer':
        prep_query = preprocessing([query])
        query_vec = tfidf_v.transform(prep_query)
        results = get_similarity(tfidf_corpus, query_vec, love_corpus)
        return results
    if model == 'BM25':
        prep_query = preprocessing([query])
        prep_query = "".join(prep_query)
        query_vec = bm25_v.transform([prep_query])
        results = get_similarity(bm25_corpus, query_vec, love_corpus)
        return results
    if model == 'Fasttext':
        prep_query = preprocessing([query])
        query_vec = fasttext(prep_query, fasttext_m)
        results = get_similarity(fasttext_corpus, query_vec, love_corpus)
        return results
    if model == 'Bert':
        query_vec = embed_bert_cls([query], bert_m, bert_t)
        results = get_similarity(bert_corpus, query_vec, love_corpus)
        return results


if __name__ == '__main__':
    final_query = st.text_input("Введи запрос ")
    final_model = st.selectbox("Выбери модель ",
                               options=['CountVectorizer', 'TfidfVectorizer', 'BM25', 'Fasttext', 'Bert'])
    number_of_docs = st.slider('Выбери количество ответов', 1, 20)
    if st.button('Найти'):
        start = time.time()
        results = search_answers(final_model, final_query)
        st.subheader('Топ-{} лучших ответов на запрос "{}":'.format(number_of_docs, final_query))
        for res in results[0:number_of_docs]:
            st.write(res)
        st.write('Time for model {}: {}'.format(final_model, time.time() - start))
