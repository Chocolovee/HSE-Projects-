import string
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import tokenize
from collections import OrderedDict
import nltk
import pymorphy2
import operator

# читаем файл
source = open('source_file.txt', "r", encoding="utf-8")
text = source.read()
source.close()
# переводим регистр
text = text.lower()

# удаляем специальные символы
spec_chars = string.punctuation + '\n\xa0«»\t—…'


# функция, которая удаляет указанный набор символов из исходного текста
def remove_chars_from_text(text, chars):
    return "".join([ch for ch in text if ch not in chars])


# удаляем спец символы и цифры из текста
text = remove_chars_from_text(text, spec_chars)
text = remove_chars_from_text(text, string.digits)

# токенизация текста
text_tokens = nltk.word_tokenize(text)

# получаем список стоп-слов
russian_stopwords = stopwords.words("russian")

# удаляем стоп слова из текста
text_tokens = [token.strip() for token in text_tokens if token not in russian_stopwords]

# нормализация
morph = pymorphy2.MorphAnalyzer()
for index, t in enumerate(text_tokens):
    text_tokens[index] = morph.parse(t)[0].normal_form

# преобразуем в текст
new_text = nltk.Text(text_tokens)

# получаем частотное распределение
fdist = FreqDist(new_text)

# вывод результата
result = open('result.txt', "w", encoding="utf-8")
for f in fdist:
    result.write(f + " " + fdist[f].__str__() + '\n')

# задание 2

# токенизация текста
text_tokens = nltk.word_tokenize(text)

# получаем список стоп-слов
russian_stopwords = stopwords.words("russian")

# удаляем стоп слова из текста
text_tokens = [token.strip() for token in text_tokens if token not in russian_stopwords]

# нормализация
stemmer = SnowballStemmer("russian")
for index, t in enumerate(text_tokens):
    text_tokens[index] = stemmer.stem(t)
# преобразуем в текст
new_text = nltk.Text(text_tokens)

# получаем частотное распределение
fdist = FreqDist(new_text)

# вывод результата
result2 = open('result2.txt', "w", encoding="utf-8")
for f in fdist:
    result2.write(f + " " + fdist[f].__str__() + '\n')

# задание 3
# используем нормализацию второго типа

# создаем новый словарь tf
tf = fdist.copy()

# считаем tf по формуле
for i in tf:
    tf[i] = tf[i] / float(len(text_tokens))

result3 = open('result3.txt', "w", encoding="utf-8")
# result3.write("словарь tf" + '\n')
# for f in tf:
#      result3.write(f + " " + tf[f].__str__() + '\n')
# result3.write('\n')

idf = fdist.copy()
for i in idf:
    idf[i] = 1

# result3.write("словарь idf" + '\n')
# for f in tf:
#      result3.write(f + " " + idf[f].__str__() + '\n')
tf_idf = fdist.copy()
for i in tf_idf:
    tf_idf[i] = tf[i] * idf[i]

tf_idf = OrderedDict(sorted(tf_idf.items(), key=lambda kv: kv[1], reverse=True))

for f in tf_idf:
    result3.write(f + " " + tf_idf[f].__str__() + '\n')

# задание 4

# читаем файл
source = open('source_file.txt', "r", encoding="utf-8")
text = source.read()
text = remove_chars_from_text(text, string.digits)

# делим на предложения
sentences = tokenize.sent_tokenize(text)


# функция получения ключа по значению
def get_key(d, value):
    for v, k in d.items():
        if v == value:
            return k
    return 0


# словарь предложений
sdict = dict()

# алгоритм
for i, s in enumerate(sentences):
    # получаем текущее предложение
    sentence = sentences[i]
    # убираем регистр
    sentence.lower()
    # удаляем числа и символы
    sentence = remove_chars_from_text(sentence, spec_chars)
    sentence = remove_chars_from_text(sentence, string.digits)
    # токенизируем предложения
    sentence_tokens = nltk.word_tokenize(sentence)
    # удаляем стопслова
    sentence_tokens = [token.strip() for token in sentence_tokens if token not in russian_stopwords]
    # нормализация с помощью стиммера
    for index, t in enumerate(sentence_tokens):
        sentence_tokens[index] = stemmer.stem(t)
    # print(sentence_tokens)
    # рассчитываем вес предложения
    sum = 0.0

    for j, t in enumerate(sentence_tokens):
        sum = sum + get_key(tf_idf, t)
    sdict[sum] = sentences[i]

# сортировка по значению
sdict_sort = OrderedDict(sorted(sdict.items(), key=lambda kv: kv[0], reverse = True))

# вывод словаря
result4 = open('result4.txt', "w", encoding="utf-8")
for f in sdict_sort:
    result4.write(f.__str__() + " " + sdict_sort[f] + " " + '\n')

# коэффициент сжатия
k = 0.2

# ищем порог веса, который отвечает за то, включаем мы предложение или нет
limit = list(sdict_sort.keys())[round((len(sdict_sort)) * 0.2)]

# вывод
for i, s in enumerate(sdict):
    if (s >= limit):
        result4.write(sdict[s] + '\n')








