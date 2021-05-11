import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)
from settings import stopwords, stopwords_eng, stem
stopwords.append('это')
stopwords.append('наш')
stopwords.append('лет')
stopwords.append('тебя')


def clear_doc(row):
    html = row['description'] + ' ' + row['key_skills']
    p = re.compile(r'<.*?>')
    text = p.sub('', html)
    return text


def get_cleared_description(kek):
    need_values = kek[['description', 'key_skills']]

    need_values['key_skills'] = need_values['key_skills'].fillna('')
    need_values['clean_doc'] = need_values.apply(clear_doc, axis=1)
    need_values['clean_doc'] = need_values['clean_doc'].apply(lambda x: x.lower())
    need_values['clean_doc'] = need_values['clean_doc'].str.replace("[^a-zа-яё+#]", " ")
    need_values['clean_doc'] = need_values['clean_doc'].apply(lambda x: ' ' + ' '.join([w for w in x.split() if len(w) > 2]))

    print('get_cleared_description end')
    return need_values


def get_cleared_key_skills(kek):
    need_values = kek['key_skills']

    need_values['key_skills'] = need_values['key_skills'].fillna('')
    need_values['clean_doc'] = need_values.apply(clear_doc, axis=1)
    need_values['clean_doc'] = need_values['clean_doc'].apply(lambda x: x.lower())
    need_values['clean_doc'] = need_values['clean_doc'].str.replace("[^a-zа-яё+#]", " ")
    need_values['clean_doc'] = need_values['clean_doc'].apply(
        lambda x: ' ' + ' '.join([w for w in x.split() if len(w) > 2]))

    print('get_cleared_description end')
    return need_values


def get_tokenized_doc(need_values):
    tokenized_doc = need_values['clean_doc'].apply(lambda x: x.split())
    stemmer = SnowballStemmer(stem)
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(
        lambda x: ' ' + ' '.join([stemmer.stem(item).lower() for item in x if stemmer.stem(item).lower() not in stopwords]))

    need_values['clean_doc'] = tokenized_doc
    print('get_tokenized_doc end')
    return need_values


def make_vectorize(need_values):
    all_stopwords = stopwords.concat(stopwords_eng)
    vectorizer = TfidfVectorizer(stop_words=all_stopwords,
                                 max_features=100,  # keep top 1000 terms
                                 max_df=0.5,
                                 smooth_idf=True)

    X = vectorizer.fit_transform(need_values['clean_doc'])
    print(X.shape, 'X shape')

    # SVD represent documents and terms in vectors
    svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100)

    svd_model.fit(X)

    print(len(svd_model.components_), '= components count')
    terms = vectorizer.get_feature_names()

    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
        print("Topic " + str(i) + ": " + ' '.join([t[0] for t in sorted_terms]))
    return X


def main():
    # kek = pd.read_csv('merged/merged.csv')
    # cleared_data = get_cleared_description(kek)
    # tokenized = get_tokenized_doc(cleared_data)
    # tokenized.to_csv('cleaned.csv', encoding='utf-8', index=False)
    tokenized = pd.read_csv('cleaned.csv')
    X = make_vectorize(tokenized)


if __name__ == '__main__':
    main()
