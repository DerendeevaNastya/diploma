import re
import warnings
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)
from settings import stopwords, stem, clusters_count, main_dict_words_count, components_count
stopwords.append('это')
stopwords.append('наш')
stopwords.append('лет')
stopwords.append('тебя')
stopwords.append('ms')


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
    print(stemmer.stem('приглашаем специалиста должность junior php разработчика требуется знание php git windows os умение работать команде'))
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(
        lambda x: ' ' + ' '.join([stemmer.stem(item).lower() for item in x if stemmer.stem(item).lower() not in stopwords]))

    need_values['clean_doc'] = tokenized_doc
    print('get_tokenized_doc end')
    return need_values


def get_all_words_dict_frequency(need_values, count_of_main_components):
    dict = {}
    for line in need_values.clean_doc:
        for word in line.split():
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 0
    kek = {k: v for k, v in sorted(dict.items(), key=lambda kv: kv[1], reverse=True)[:count_of_main_components]}
    print(kek)
    print('length of dictionary is: ', len(kek))
    return kek


def get_all_words_by_docs_count_dict(need_values, words):
    dict = {word: 0 for word in words}
    for line in need_values.clean_doc:
        kek = line.split()
        for word in words:
            if word in kek:
                dict[word] += 1

    return dict


def make_tf_idf_matrix(need_values, main_dict, main_dict_by_docs):
    kek = []
    all_count = []
    freq_log = np.hstack([len(need_values.clean_doc) / main_dict_by_docs[word] for word in main_dict])
    for line in need_values.clean_doc:
        freq = {k: 0 for k in main_dict}
        for word in line.split():
            if word in freq:
                freq[word] += 1

        main_sum = sum(freq.values())
        if main_sum == 0:
            continue
        all_count.append(main_sum)
        kek.append(list(freq.values()))
    return np.vstack(kek) * np.log(freq_log)


def get_squared_evcklid_dist(row1, row2):
    return np.sum((row1 - row2) ** 2)


def get_sorted_words_by_evcklid(main_dict_words, component_labels, centres, words_count, rows):
    all = []
    for i in range(clusters_count):
        d = {main_dict_words[j]: rows[j] for j in range(len(component_labels)) if component_labels[j] == i}
        s = sorted(d.items(), key=lambda x: get_squared_evcklid_dist(x[1], centres[i]))
        mapped = [get_squared_evcklid_dist(x[1], centres[i]) for x in s]
        print(' '.join([elem[0] for elem in s][:words_count]))
        # print(mapped[:words_count])
        all.append([elem[0] for elem in s][:words_count])
    return all


def get_words_lists(main_dict_words, component_labels):
    all = []
    for i in range(clusters_count):
        d = [main_dict_words[j] for j in range(len(component_labels)) if component_labels[j] == i]
        print("dictionary words count")
        print(d)
        all.append(d)
    return all


def make_k_means_clusterization(X, n_components):
    kmeans = KMeans(n_clusters=n_components, random_state=0).fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


def filter_by_labels(x, y, val, w_labels):
    x1 = np.hstack(x)
    y1 = np.hstack(y)
    return (np.vstack([x1[i] for i, v in enumerate(w_labels) if v == val]),
            np.vstack([y1[i] for i, v in enumerate(w_labels) if v == val]))


def make_vectorize(need_values, max_features, n_iter):
    main_dict = get_all_words_dict_frequency(need_values, max_features)
    main_dict_by_docs = get_all_words_by_docs_count_dict(need_values, list(main_dict.keys()))
    tf_idf = make_tf_idf_matrix(need_values, main_dict, main_dict_by_docs)
    svd_model = TruncatedSVD(n_components=components_count, n_iter=n_iter)

    kek = svd_model.fit_transform(tf_idf.T)
    print(kek.shape)
    print()
    labels, centres = make_k_means_clusterization(kek, clusters_count)
    get_sorted_words_by_evcklid(list(main_dict.keys()), labels, centres, 10, kek)
    print('\n' * 3)

    print(len(svd_model.components_), '= components count')
    terms = main_dict.keys()
    print(len(terms))

    return centres


def get_sorted_by_evclid_all_centres(centres, vectors):
    kek = []
    for vector in vectors:
        all = []
        for center in centres:
            all.append(get_squared_evcklid_dist(center, vector))
        kek.append(all)
    return kek


def main():
    # kek = pd.read_csv('merged/merged.csv')
    # cleared_data = get_cleared_key_skills(kek)
    # tokenized = get_tokenized_doc(cleared_data)
    # tokenized.to_csv('cleaned_key_skills_without_ms.csv', encoding='utf-8', index=False)
    tokenized = pd.read_csv('cleaned_key_skills_without_ms.csv')
    print(tokenized.shape)
    tokenized = tokenized.dropna()
    print(tokenized.shape)

    make_vectorize(tokenized, main_dict_words_count, 30)


if __name__ == '__main__':
    main()
