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
    html = ' ' + row['key_skills']
    p = re.compile(r'<.*?>')
    text = p.sub('', html)
    return text


def full_for_tf_correct(row):
    all = row.clean_doc.split()
    line_for_tf = 'fuck ' * 10
    if len(all) < 3:
        return row.clean_doc + line_for_tf
    return row.clean_doc


def get_cleared_key_skills(kek):
    need_values = kek[['key_skills']]
    print(need_values)
    need_values.key_skills = need_values.key_skills.fillna('')
    need_values['clean_doc'] = need_values.apply(clear_doc, axis=1)
    need_values['clean_doc'] = need_values['clean_doc'].apply(lambda x: x.lower())
    need_values['clean_doc'] = need_values['clean_doc'].str.replace("[^a-zа-яё+#]", " ")
    need_values['clean_doc'] = need_values['clean_doc'].apply(
        lambda x: ' ' + ' '.join([w for w in x.split() if len(w) >= 2]))

    print('get_cleared_description end')
    return need_values


def get_tokenized_doc(need_values):
    # need_values.clean_doc = need_values.apply(full_for_tf_correct, axis=1)
    tokenized_doc = need_values['clean_doc'].apply(lambda x: x.split())
    print(need_values.shape)
    need_values = need_values.dropna()
    print(need_values.shape)
    stemmer = SnowballStemmer(stem)
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(
        lambda x: ' ' + ' '.join([stemmer.stem(item).lower() for item in x if stemmer.stem(item).lower() not in stopwords]))

    need_values['clean_doc'] = tokenized_doc
    print('get_tokenized_doc end')
    return need_values


def make_vectorize(need_values, max_features, max_df, n_iter):
    all_stopwords = stopwords + stopwords_eng
    vectorizer = TfidfVectorizer(max_features=max_features,  # keep top 1000 terms
                                 max_df=max_df,
                                 smooth_idf=True)

    X = vectorizer.fit_transform(need_values['clean_doc'])
    print(X.shape, 'X shape')
    print(X)


    # SVD represent documents and terms in vectors
    svd_model = TruncatedSVD(n_components=20, algorithm='arpack', n_iter=n_iter)

    svd_model.fit(X)

    print(len(svd_model.components_), '= components count')
    terms = vectorizer.get_feature_names()
    print(terms)

    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
        # print(sorted_terms)
        print("Topic " + str(i) + ": " + ' '.join([t[0] for t in sorted_terms]))
        print('Weigh ' + str( ) + ': ' + ' '.join([str(t[1]) for t in sorted_terms]))
    return X


def main():
    # kek = pd.read_csv('merged/merged.csv')
    # cleared_data = get_cleared_key_skills(kek)
    # tokenized = get_tokenized_doc(cleared_data)
    # tokenized.to_csv('cleaned_key_skills_fuck.csv', encoding='utf-8', index=False)
    tokenized = pd.read_csv('cleaned_key_skills_fuck.csv')
    # print(tokenized.key_skills[:15])
    print('!\n'.join([i for i in tokenized.clean_doc[:10000] if 'sap' in i]))
    # print('!!!!')
    # for max_iter in [100, 300!, 500]:
    #     print('\n' * 8)
    #     for max_features in [100, 200, 300!, 400, 700, 1000]:
    #         print('\n' * 3)
    #         for max_idf in [0.1, 0.3, 0.5!, 0.8, 1]:
    #             make_vectorize(tokenized, max_features, max_idf, max_iter)

    make_vectorize(tokenized, 300, 0.5, 10)


if __name__ == '__main__':
    main()
