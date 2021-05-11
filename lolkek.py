#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
# https://habr.com/ru/post/323516/

from tkinter.filedialog import *
from tkinter.messagebox import *
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import nltk
# -- код для загрузки стоп-слов и знаков препинания
import scipy
# nltk.download('stopwords')
# nltk.download('punkt')
from settings import docs, stem, stopwords, clusters_count, clr
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import sys

stemmer = SnowballStemmer(stem)
doc = [w for w in docs]
ddd = len(docs)


def get_texts_from_dataframe():
    tokenized = pd.read_csv('cleaned_key_skills_fuck.csv')
    return list(tokenized.clean_doc)


# docs = [w for w in get_texts_from_dataframe()]
# doc = [w for w in get_texts_from_dataframe()]


def start():
    clear_all()
    txt.insert(END, 'Исходные документы\n')
    for k, v in enumerate(docs):
        txt.insert(END, 'Ном.док--%u Текст-%s \n' % (k, v))
    if var1.get() == 0:
        return word_1()
    elif var1.get() == 1:
        t = " "
        word = nltk.word_tokenize((' ').join(doc))
        stopword = [stemmer.stem(w).lower() for w in stopwords]
        return WordStopDoc(t, stopword)


def word_1():
    txt1.delete(1.0, END)
    txt2.delete(1.0, END)
    word = nltk.word_tokenize((' ').join(doc))
    n = [stemmer.stem(w).lower() for w in word if len(w) > 1 and w.isalpha()]
    stopword = [stemmer.stem(w).lower() for w in stopwords]
    fdist = nltk.FreqDist(n)
    t = fdist.hapaxes()
    txt1.insert(END, 'Слова которые встречаются только один раз:\n%s' % t)
    txt1.insert(END, '\n')
    return WordStopDoc(t, stopword)


def WordStopDoc(t, stopword):
    d = {}
    c = []
    p = {}
    for i in range(0, len(doc)):
        word = nltk.word_tokenize(doc[i])
        word_stem = [stemmer.stem(w).lower() for w in word if len(w) > 1 and w.isalpha()]
        word_stop = [w for w in word_stem if w not in stopword]
        words = [w for w in word_stop if w not in t]
        p[i] = [w for w in words]
        for w in words:
            if w not in c:
                c.append(w)
                d[w] = [i]
            elif w in c:
                d[w] = d[w] + [i]
    txt1.insert(END, 'Стоп-слова:\n')
    txt1.insert(END, stopwords)
    txt1.insert(END, '\n')
    txt1.insert(END, 'Cлова(основа):\n')
    txt1.insert(END, c)
    txt1.insert(END, '\n')
    txt1.insert(END, ' Распределение слов по документам:\n')
    txt1.insert(END, d)
    txt1.insert(END, '\n')
    return Create_Matrix(d, c, p)


def Create_Matrix(d, c, p):
    a = len(c)
    b = len(doc)
    A = np.zeros([a, b])
    c.sort()
    for i, k in enumerate(c):
        for j in d[k]:
            A[i, j] += 1
    txt1.insert(END, 'Первая матрица для проверки заполнения строк и столбцов:\n')
    txt1.insert(END, A)
    txt1.insert(END, '\n')
    return Analitik_Matrix(A, c, p)


def Analitik_Matrix(A, c, p):
    wdoc = np.sum(A, axis=0)
    pp = []
    q = -1
    for w in wdoc:
        q = q + 1
        if w == 0:
            pp.append(q)
    if len(pp) != 0:
        for k in pp:
            doc.pop(k)
        word_1()
    elif len(pp) == 0:
        rows, cols = A.shape
        txt1.insert(END, 'Исходная частотная матрица число слов---%u больше либо равно числу документов-%u \n' % (
        rows, cols))
        nn = []
        for i, row in enumerate(A):
            st = (c[i], row)
            stt = sum(row)
            nn.append(stt)
            txt1.insert(END, st)
            txt1.insert(END, '\n')
        if var.get() == 0:
            return TF_IDF(A, c, p)
        elif var.get() == 1:
            l = nn.index(max(nn))
            return U_S_Vt(A, c, p, l)


def TF_IDF(A, c, p):
    wpd = np.sum(A, axis=0)
    dpw = np.sum(np.asarray(A > 0, 'i'), axis=1)
    rows, cols = A.shape
    txt1.insert(END,
                'Нормализованная по методу TF-IDF матрица: строк- слов -%u столбцов - документов--%u \n' % (rows, cols))
    for i in range(rows):
        for j in range(cols):
            m = np.float(A[i, j]) / wpd[j]
            n = np.log(np.float(cols) / dpw[i])
            A[i, j] = np.round(n * m, 2)
    gg = []
    for i, row in enumerate(A):
        st = (c[i], row)
        stt = sum(row)
        gg.append(stt)
        txt1.insert(END, st)
        txt1.insert(END, '\n')
    l = gg.index(max(gg))
    return U_S_Vt(A, c, p, l)


def U_S_Vt(A, c, p, l):
    print(A.shape, '!!!!!!!!!!!!!!!!!!!!')
    U, S, Vt = np.linalg.svd(A)
    rows, cols = U.shape
    for j in range(0, cols):
        for i in range(0, rows):
            U[i, j] = round(U[i, j], 4)
    txt1.insert(END,
                ' Первые 2 столбца ортогональной матрицы U слов, сингулярного преобразования нормализованной матрицы: строки слов -%u\n' % rows)
    for i, row in enumerate(U):
        st = (c[i], row[0:2])
        txt1.insert(END, st)
        txt1.insert(END, '\n')
    kt = l
    wordd = c[l]
    res1 = -1 * U[:, 0:1]
    wx = res1[kt]
    res2 = -1 * U[:, 1:2]
    wy = res2[kt]
    txt1.insert(END, ' Координаты x --%f и y--%f опорного слова --%s, от которого отсчитываются все расстояния \n' % (
    wx, wy, wordd))
    txt1.insert(END, ' Первые 2 строки диагональной матрица S \n')
    Z = np.diag(S)
    txt1.insert(END, Z[0:2, 0:2])
    txt1.insert(END, '\n')
    rows, cols = Vt.shape
    for j in range(0, cols):
        for i in range(0, rows):
            Vt[i, j] = round(Vt[i, j], 4)
    txt1.insert(END,
                ' Первые 2 строки ортогональной матрицы Vt документов сингулярного преобразования нормализованной матрицы: столбцы документов -%u\n' % cols)
    st = (-1 * Vt[0:2, :])
    txt1.insert(END, st)
    txt1.insert(END, '\n')
    res3 = (-1 * Vt[0:1, :])
    res4 = (-1 * Vt[1:2, :])
    X = np.dot(U[:, 0:2], Z[0:2, 0:2])
    Y = np.dot(X, Vt[0:2, :])
    txt1.insert(END, ' Матрица для выявления скрытых связей \n')
    rows, cols = Y.shape
    for j in range(0, cols):
        for i in range(0, rows):
            Y[i, j] = round(Y[i, j], 2)
    for i, row in enumerate(Y):
        st = (c[i], row)
        txt1.insert(END, st)
        txt1.insert(END, '\n')
    return Word_Distance_Document(res1, wx, res2, wy, res3, res4, Vt, p, c, U)


labels = []


def Word_Distance_Document(res1, wx, res2, wy, res3, res4, Vt, p, c, U):
    xx, yy = -1 * Vt[0:2, :]
    rows, cols = Vt.shape
    a = cols
    b = cols
    B = np.zeros([a, b])
    for i in range(0, cols):
        for j in range(0, cols):
            xxi, yyi = -1 * Vt[0:2, i]
            xxi1, yyi1 = -1 * Vt[0:2, j]
            B[i, j] = round(
                float(xxi * xxi1 + yyi * yyi1) / float(np.sqrt((xxi * xxi + yyi * yyi) * (xxi1 * xxi1 + yyi1 * yyi1))),
                6)
    txt1.insert(END, ' Матрица косинусных расстояний между документами\n')
    txt1.insert(END, B)
    txt1.insert(END, '\n')
    txt1.insert(END, ' Кластеризация косинусных расстояний между документами\n')
    X = np.array(B)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    txt1.insert(END, 'Метки кластеров\n')
    txt1.insert(END, kmeans.labels_)
    txt1.insert(END, '\n')
    txt1.insert(END, 'Координаты центроидов кластеров\n')
    txt1.insert(END, kmeans.cluster_centers_)
    txt1.insert(END, '\n')
    Q = U
    UU = Q.T
    rows, cols = UU.shape
    a = cols
    b = cols
    B = np.zeros([a, b])
    for i in range(0, cols):
        for j in range(0, cols):
            xxi, yyi = -1 * UU[0:2, i]
            xxi1, yyi1 = -1 * UU[0:2, j]
            if float(xxi * xxi1 + yyi * yyi1) < sys.float_info.epsilon:
                B[i, j] = 0
            else:
                B[i, j] = round(float(xxi * xxi1 + yyi * yyi1) / float(
                    np.sqrt((xxi * xxi + yyi * yyi) * (xxi1 * xxi1 + yyi1 * yyi1))), 6)
    txt1.insert(END, ' Матрица косинусных расстояний между словами\n')
    for i, row in enumerate(B):
        st = (c[i], row[0:])
        txt1.insert(END, st)
        txt1.insert(END, '\n')
    txt1.insert(END, ' Кластеризация косинусных расстояний между словами\n')
    X = np.array(B)
    kmeans = KMeans(n_clusters=clusters_count, random_state=0).fit(X)
    txt1.insert(END, 'Метки клайстеров\n')
    txt1.insert(END, kmeans.labels_)
    print(kmeans.labels_.shape)
    txt1.insert(END, '\n')
    txt1.insert(END, ' Координаты центроидов кластеров\n')
    txt1.insert(END, kmeans.cluster_centers_)
    arts = []
    txt2.insert(END,
                'Результаты анализа: Всего документов:%u. Осталось документов после исключения не связанных:%u\n' % (
                ddd, len(doc)))
    if ddd > len(doc):
        txt2.insert(END, " Оставшиеся документы после исключения не связанных:")
        txt2.insert(END, '\n')
        for k, v in enumerate(doc):
            ww = 'Док.№ - %i. Text -%s' % (k, v)
            txt2.insert(END, ww)
            txt2.insert(END, '\n')
    for k in range(0, len(doc)):
        ax, ay = xx[k], yy[k]
        dx, dy = float(wx - ax), float(wy - ay)
        if var2.get() == 0:
            dist = float(np.sqrt(dx * dx + dy * dy))
        elif var2.get() == 1:
            dist = float(wx * ax + wy * ay) / float(np.sqrt(wx * wx + wy * wy) * np.sqrt(ax * ax + ay * ay))
        arts.append((k, p[k], round(dist, 3)))
    q = (sorted(arts, key=lambda a: a[2]))
    dd = []
    ddm = []
    aa = []
    bb = []
    for i in range(1, len(doc)):
        cos1 = q[i][2]
        cos2 = q[i - 1][2]
        if var2.get() == 0:
            qq = round(float(cos1 - cos2), 3)
        elif var2.get() == 1:
            sin1 = np.sqrt(1 - cos1 ** 2)
            sin2 = np.sqrt(1 - cos2 ** 2)
            qq = round(float(1 - abs(cos1 * cos2 + sin1 * sin2)), 3)
        tt = [(q[i - 1])[0], (q[i])[0]]
        dd.append(tt)
        ddm.append(qq)
    for w in range(0, len(dd)):
        i = ddm.index(min(ddm))
        aa.append(dd[i])
        bb.append(ddm[i])
        del dd[i]
        del ddm[i]
    for i in range(0, len(aa)):
        if len([w for w in p[aa[i][0]] if w in p[aa[i][1]]]) != 0:
            zz = [w for w in p[aa[i][0]] if w in p[aa[i][1]]]
        else:
            zz = ['нет общих слов']
        cs = []
        for w in zz:
            if w not in cs:
                cs.append(w)
        if var2.get() == 0:
            sc = "Евклидова мера расстояния "
        elif var2.get() == 1:
            sc = "Косинусная мера расстояния "
        tr = '№№ Док %s- %s-%s -Общие слова -%s' % (aa[i], bb[i], sc, cs)
        txt2.insert(END, tr)
        txt2.insert(END, '\n')
    return Grafics_End(res1, res2, res3, res4, kmeans.labels_, c)


def filter_by_labels(x, y, val, w_labels):
    x1 = np.hstack(x)
    y1 = np.hstack(y)
    print(labels)
    return (np.vstack([x1[i] for i, v in enumerate(w_labels) if v == val]),
            np.vstack([y1[i] for i, v in enumerate(w_labels) if v == val]))


def Grafics_End(res1, res2, res3, res4, word_labels, c):  # Построение график с программным управлением масштабом
    plt.title('Semantic space', size=14)
    plt.xlabel('x-axis', size=14)
    plt.ylabel('y-axis', size=14)
    e1 = (max(res1) - min(res1)) / len(c)
    e2 = (max(res2) - min(res2)) / len(c)
    e3 = (max(res3[0]) - min(res3[0])) / len(doc)
    e4 = (max(res4[0]) - min(res4[0])) / len(doc)
    print(res1.shape)
    plt.axis([min(res1) - e1, max(res1) + e1, min(res2) - e2, max(res2) + e2])
    for i in range(clusters_count):
        x1, y1 = filter_by_labels(res1, res2, i, word_labels)
        plt.plot(x1, y1, color=clr[i % clusters_count], linestyle=' ', marker='.', ms=10, label='Words' + str(i))
    # plt.axis([min(res3[0])-e3, max(res3[0])+e3, min(res4[0])-e4, max(res4[0])+e4])
    # plt.plot(res3[0], res4[0], color='b', linestyle=' ', marker='o',ms=10,label='Documents №')

    plt.legend(loc='best')
    k = {}
    for i in range(0, len(res1)):
        xv = float(res1[i])
        yv = float(res2[i])
        if (xv, yv) not in k.keys():
            k[xv, yv] = c[i]
        elif (xv, yv) in k.keys():
            k[xv, yv] = k[xv, yv] + ',' + c[i]
        plt.annotate(k[xv, yv], xy=(res1[i], res2[i]), xytext=(res1[i], res2[i]))
    k = {}
    for i in range(0, len(doc)):
        xv = float((res3[0])[i])
        yv = float((res4[0])[i])
        if (xv, yv) not in k.keys():
            k[xv, yv] = str(i)
        elif (xv, yv) in k.keys():
            k[xv, yv] = k[xv, yv] + ',' + str(i)
        # plt.annotate(k[xv,yv], xy=((res3[0])[i], (res4[0])[i]), xytext=((res3[0])[i], (res4[0])[i]))
    plt.grid()
    plt.show()


def close_win():
    if askyesno("Exit", "Do you want to quit?"):
        tk.destroy()


def save_text():
    save_as = asksaveasfilename()
    try:
        x = txt.get(1.0, END) + '\n' * 2 + txt1.get(1.0, END) + '\n' * 2 + txt2.get(1.0, END)
        f = open(save_as, "w")
        f.writelines(x)
        f.close()
    except:
        pass
    clear_all()


def clear_all():
    txt.delete(1.0, END)
    txt1.delete(1.0, END)
    txt2.delete(1.0, END)


tk = Tk()
tk.geometry('700x650')
main_menu = Menu(tk)
tk.config(menu=main_menu)
file_menu = Menu(main_menu)
main_menu.add_cascade(label="LSA", menu=file_menu)
file_menu.add_command(label="Start", command=start)
file_menu.add_command(label="Save text", command=save_text)
file_menu.add_command(label="Clear all fields", command=clear_all)
file_menu.add_command(label="Exit", command=close_win)
txt = Text(tk, width=72, height=10, font="Arial 12", wrap=WORD)
txt.pack()
txt1 = Text(tk, width=72, height=10, font="Arial 12", wrap=WORD)
txt1.pack()
txt2 = Text(tk, width=72, height=10, font="Arial 12", wrap=WORD)
txt2.pack()
var = IntVar()
ch_box = Checkbutton(tk, text="no to use TF_IDF", variable=var)
ch_box.pack()
var1 = IntVar()
ch_box1 = Checkbutton(tk, text="no to exclude words used once", variable=var1)
ch_box1.pack()
var2 = IntVar()
ch_box2 = Checkbutton(tk, text="Evckid distance/cos distance", variable=var2)
ch_box2.pack()
tk.title("System of the automated semantic analysis")
tk.mainloop()
