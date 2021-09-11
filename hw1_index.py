import os
import re
import nltk
import pandas as pd
import pymorphy2
import operator
from pymystem3 import Mystem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')
stopword = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
m = Mystem()


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
    for k, v in eps.items():
        with open(v + '/' + k, 'r', encoding='utf-8-sig') as f:
            text = f.read()
            texts.append(text)
    return texts


"""
я пробовала pymorphy для лематизации, с ним получаются другой набор слов, встречающийся во всех текстах и другие
количества упоминаний героев (из-за качества парсеров). Я оставила в итоговой версии mystem по следующим причинам:
парсер работает быстрее, лучше лемматизирует слова (например, pymorphy лемматизировал Джоуи как Джоуя).
"""


def preprocessing(texts):
    corpus = []
    for onestr in texts:
        onestr = re.sub(r'[0-9]', '', onestr)
        onestr = re.sub(r'[a-zA-Z]', '', onestr)
        tokens = nltk.word_tokenize(onestr)
        tokens = ' '.join(tokens)
        lemmatized_text = m.lemmatize(tokens)
        for word in lemmatized_text:
            if word in stopword:
                lemmatized_text.remove(word)
        lemmatized_text = ' '.join(lemmatized_text)
        corpus.append(lemmatized_text)
    return corpus


"""
к 165-ти текстам я добавила еще одну строку, которая отображает суммарное количество встречаемости слов в корпусе.
"""


def term_document(corpus):
    vectorizer = CountVectorizer(analyzer='word')
    x = vectorizer.fit_transform(corpus)
    data = pd.DataFrame(x.toarray())
    data.columns = vectorizer.get_feature_names()
    data = data.append(data.sum(axis=0), ignore_index=True)
    return data


"""
# слов, которые встречаются только 1 раз за все время много, так что программа выводит рандомно одно из них.
"""


def statistics(df):
    max_and_min = df.sort_values(by=165, axis=1, ascending=False).iloc[:, list(range(1)) + [-1]]
    popular = 'самое частое слово: ' + list(max_and_min.columns)[0]
    unique = 'самое редкое слово: ' + list(max_and_min.columns)[1]
    df_presence = pd.DataFrame([df.astype(bool).sum(axis=0)])
    everywhere = []
    for name in df_presence:
        if df_presence[name].values[0] == 166:
            everywhere.append(name)
    popularity = {'Моника': 0, 'Рейчел': 0, 'Чендлер': 0, 'Фиби': 0, 'Росс': 0, 'Джоуи': 0}
    for name in df:
        if name == 'моника' or name == 'мон':
            popularity['Моника'] += df[name].values[-1]
        if name == 'рейчел' or name == 'рейч' or name == 'рэйч' or name == 'рэйчел':
            popularity['Рейчел'] += df[name].values[-1]
        if name == 'чендлер' or name == 'чэндлер' or name == 'чен' or name == 'чэн':
            popularity['Чендлер'] += df[name].values[-1]
        if name == 'фиби' or name == 'фибс':
            popularity['Фиби'] += df[name].values[-1]
        if name == 'росс':
            popularity['Росс'] += df[name].values[-1]
        if name == 'джоуи' or name == 'джои' or name == 'джо':
            popularity['Джоуи'] += df[name].values[-1]
    return popular, unique, everywhere, popularity


if __name__ == '__main__':
    corp = preprocessing(get_texts('friends-data-2'))
    results = statistics(term_document(corp))
    for item in results[0:2]:
        print(item)
    print('слова, встречающиеся во всех документах:', results[2])
    print('самый популярный герой: ', max(results[3].items(), key=operator.itemgetter(1))[0])
