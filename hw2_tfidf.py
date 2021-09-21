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


"""
foldername - название папки, в которой лежат папки со всеми сезонами сериала. У меня она названа как 'friends-data-2'
(см. конец кода).
"""


def get_texts(foldername):
    curr_dir = os.getcwd()
    filepath = os.path.join(curr_dir, foldername)
    files = list(os.walk(filepath))
    friends = []
    for i, element in enumerate(files):
        if i != 0:
            friends.append(element)
    eps = {}
    for season in friends:
        for episode in season[2]:
            eps[episode] = season[0]
    texts = []
    titles = []
    for k, v in eps.items():
        with open(v + '/' + k, 'r', encoding='utf-8-sig') as f:
            text = f.read()
            texts.append(text)
            titles.append(k)
    return texts, titles


def preprocessing(texts):
    corpus = []
    for text in tqdm(texts):
        text = str(text)
        tokens = m.lemmatize(text.lower())
        tokens = [token for token in tokens if token not in stopword
                  and token != " "
                  and len(token) >= 3
                  and token.strip() not in punctuation
                  and token.isdigit() == False]
        text = " ".join(tokens)
        text = re.sub(r'[a-zA-Z]', '', text)
        corpus.append(text)
    return corpus


def req_processing(request):
    req = str(request)
    tokens = m.lemmatize(req.lower())
    tokens = [token for token in tokens if token not in stopword
            and token != " "
            and len(token) >= 3
            and token.strip() not in punctuation
            and token.isdigit() == False]
    req = " ".join(tokens)
    req = re.sub(r'[a-zA-Z]', '', req)
    vectreq = vectorizer.transform([req]).toarray()
    return vectreq


def get_cosine(vectcorp, vectreq):
    cdist = vectcorp.dot(vectreq.transpose())
    return cdist


if __name__ == '__main__':
    vectorizer = TfidfVectorizer()
    corpus = preprocessing(get_texts('friends-data-2')[0])
    vectcorp = vectorizer.fit_transform(corpus).toarray()
    while True:
        request = input("введите запрос: ")
        vectreq = req_processing(request)
        cdist = get_cosine(vectcorp, vectreq)
        onecdist = cdist.flatten()
        order = sorted(range(len(onecdist)), key=lambda k: onecdist[k], reverse=True)
        titles = get_texts('friends-data-2')[1]
        for item in order:
            print(titles[item])
