#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
from nltk import *
from nltk.corpus import brown
stopwords= nltk.corpus.stopwords.words('russian')
stopwords_eng = nltk.corpus.stopwords.words('english')
docs =[
    "Британская полиция знает о местонахождении основателя WikiLeaks",# Документ № 0
    "В суде США начинается процесс против россиянина, рассылавшего спам",# Документ №1
    "Церемонию вручения Нобелевской премии мира бойкотируют 19 стран",# Документ №2
    "В Великобритании арестован основатель сайта Wikileaks Джулиан Ассандж",# Документ №3
    "Украина игнорирует церемонию вручения Нобелевской премии",# Документ №4
    "Шведский суд отказался рассматривать апелляцию основателя Wikileaks",# Документ №5
    "НАТО и США разработали планы обороны стран Балтии против России",# Документ №6
    "Полиция Великобритании нашла основателя WikiLeaks, но, не арестовала",# Документ №7
    "В Стокгольме и Осло сегодня состоится вручение Нобелевских премий"# Документ №8
 ]
stem = 'russian'
clusters_count = 20
components_count = 10
main_dict_words_count = 300
clr = ['lightblue', 'lightgreen', 'lime', 'magenta', 'maroon', 'navy', 'olive', 'yellow', 'orchid', 'orange', 'pink', 'plum', 'purple', 'red', 'salmon']
#'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian',
#'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish'