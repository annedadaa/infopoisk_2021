import os
import re
import nltk
import pymorphy2
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
stopword = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
m = Mystem()
punctuation += '...'
vectorizer = TfidfVectorizer()


def get_paths(foldername):
    curr_dir = os.getcwd()
    datapath = os.path.join(curr_dir, foldername)
    filepaths = []
    titles = []
    for path, dirs, files in os.walk(datapath):
        for name in files:
            filepaths.append(os.path.join(path, name))
            titles.append(name)
    filepaths = filepaths[1:]
    titles = titles[1:]
    return filepaths, titles


def get_texts(filepaths):
    texts = []
    for file in filepaths:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        texts.append(text)
    return texts


def preprocessing(texts):
    corpus = []
    for text in tqdm(texts):
        text = str(text)
        tokens = m.lemmatize(text.lower())
        tokens = [token for token in tokens if token not in stopword
                  and token != " "
                  and len(token) >= 3
                  and token.strip() not in punctuation
                  and token.isdigit() is False]
        text = " ".join(tokens)
        text = re.sub(r'[a-zA-Z]', '', text)
        corpus.append(text)
    return corpus


def preprocessing_request(request):
    tokens = m.lemmatize(request.lower())
    tokens = [token for token in tokens if token not in stopword
              and token != " "
              and len(token) >= 3
              and token.strip() not in punctuation
              and token.isdigit() is False]
    req = re.sub(r'[a-zA-Z]', '', " ".join(tokens))
    vrequest = vectorizer.transform([req]).toarray()
    return vrequest


def get_cosine(vcorpus, vrequest):
    cosine = vcorpus.dot(vrequest.transpose())
    return cosine


if __name__ == '__main__':
    paths, eps = get_paths('friends-data')
    corp = preprocessing(get_texts(paths))
    vectorizedcorpus = vectorizer.fit_transform(corp).toarray()
    while True:
        yourtext = input("введите запрос: ")
        vtext = preprocessing_request(yourtext)
        cos = get_cosine(vectorizedcorpus, vtext)
        order = sorted(range(len(cos.flatten())), key=lambda k: cos.flatten()[k], reverse=True)
        for item in order:
            print(eps[item])
