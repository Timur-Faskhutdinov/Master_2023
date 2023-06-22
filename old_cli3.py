__author__ = "Faskhutdinov Timur"
#region Imports
#################################################################
# Textblob
from textblob import TextBlob
from textblob import Word
from textblob import WordList
import nltk
nltk.download('punkt')

from collections import Counter
import re
import json

# Расстояние Левенштейна
from Levenshtein import distance

# Обработка pdf-файлов
import PyPDF2 as pdf2

import cli
#################################################################
#endregion

#region Configs
#################################################################
N_NUMBER = 3
OCCURRENCE_THRESHOLD = 1
SUBSTITUTE_SYMBOL = 'ƪ'
PREPROCESSING_PATTERN = re.compile(r'[^а-яА-Я\s\-ƪ]')
#################################################################
#endregion

#region Генерация словаря устойчивых конструкций
#################################################################
def text_preprocessing(text):   # предобработка текста
  # old_pattern = r'[—−§«»•]'
  text = re.sub(PREPROCESSING_PATTERN, '', text)
  return text

def add_frequent_wordings(text, counter, N_NUMBER):             # обработка страницы
  text = text_preprocessing(text)
  inputBlob = TextBlob(text)
  inputNgramm = list(map(tuple, inputBlob.ngrams(n=N_NUMBER)))  # построение n-грамм
  input_counter = Counter(inputNgramm)                          # подсчет количества вхождений
  counter.update(input_counter)


def process_frequent_wording_pdf(file_path, wording_counter, N_NUMBER):   # обработка pdf-файла
    with open (file_path, "rb") as f:
      pdf = pdf2.PdfReader(f)
      for page in pdf.pages:
        add_frequent_wordings(page.extract_text(), wording_counter, N_NUMBER)


def process_frequent_wording_files(file_path_list, N_NUMBER): # функция генерации
                                                              # словаря устойчивых конструкций
  counter = Counter({})
  for path in file_path_list:
    process_frequent_wording_pdf(path, counter, N_NUMBER)

  frequentWordingDict = {key:value for key,value in counter.items()
    if (value > OCCURRENCE_THRESHOLD) and   # фильтрация недостаточно часто встречающихся конструкций
      (any(len(word) > 1 for word in key))} # фильтация конструкций,
                                            # полностью состоящих из односимвольных слов
  return frequentWordingDict


def serialize_dict(dict_path, dict_content):
  # Конвертирование словаря в лист кортежей,
  # т.к. json не поддерживает словари с ключами-кортежами
  packed_data = [(key, value) for key,value in dict_content.items()]
  with open(dict_path, "w") as f:
    f.write(json.dumps(packed_data))


def deserialize_dict(dict_path):
  with open(dict_path, "r") as f:
    file_content = f.read()
  packed_data = json.loads(file_content)
  return {tuple(key):value for key,value in packed_data}
#################################################################
#endregion

#region Поиск в словаре устойчивых конструкций
#################################################################
def search_in_dict(text, wdict, N_NUMBER):
  text = text_preprocessing(text)
  searchBlob = TextBlob(text)

  # Поиск поврежденных слов
  damagedWords = {}
  k=0
  for word in searchBlob.words:
    if(word.find(SUBSTITUTE_SYMBOL) != -1):
      damagedWords[k]=[word, ""] # поврежденная и исправленная формы
    k+=1

  # Формирование n-грамм для поиска, n-граммы сгруппированы по повреденному слову
  # Каждую группу формируют n-граммы, содержащие поврежденное слово
  searchNgramms = {}
  for key,value in damagedWords.items():
    tempList = []
    for i in range(N_NUMBER):
      # Проверки на выход за границы списка
      if(key - N_NUMBER + 1 + i < 0): continue
      if(key + i) >= k: break
      tempList.append(tuple(searchBlob.words[key-N_NUMBER+1+i : key+1+i]))
    searchNgramms[key]=list(tempList) # ключ - порядковый номер слова в тексте

  # Поиск в словаре устойчивых конструкций
  for key,value in searchNgramms.items():
    for search_n in value:
      i=[]  # индексы поврежденных слов - будет сравниваться в последнюю очередь
      eq = list(wdict.keys())  # совпадающие устойчивые конструкции

      for word, k in zip(search_n, range(len(search_n))):
        if(word.find(SUBSTITUTE_SYMBOL) != -1):
          i.append(k)
          continue
        # Сравниваем слова по порядку с n-граммами словаря (ленивые вычисления)
        eq = filter(lambda nGramm: nGramm[k] == word, eq)

      # Сравниваем поврежденное слово
      for index in i:
        subst_number = search_n[index].count(SUBSTITUTE_SYMBOL) # число поврежденных символов в слове
        # Фильтруем по расстоянию Левенштейна == числу поврежденных символов в слове
        eq = filter(lambda nGramm: distance(nGramm[index], search_n[index]) == subst_number, eq)

      eqList = list(eq) # Ленивые фильтрации вычисляются здесь
      if(len(eqList) != 0):
        for index in i:   # Ищем восстанавливаемое слово (восстанавливается 1 слово за итерацию)
          if(damagedWords[key][0] == search_n[index]):
            # Выбирается наиболее часто встречаемая устойчивая конструкция
            damagedWords[key][1] = max(eqList, key=lambda n: wdict[n])[index]
            break

  return damagedWords
#################################################################
#endregion

#region Консольное приложение
#################################################################
app = cli.App(
  name='n-rec',
  usage='',
  version='0.1',
)

@app.command()
def generate(ctx: cli.Context, inp: str = '', outp: str = '', n: int = N_NUMBER, t: int = OCCURRENCE_THRESHOLD):
    """
    Генерация словаря устойчивых конструкций.

    :param inp: файлы текстов-доноров.
        Перечислять через запятую без пробелов!
    
    :param outp: путь, куда будет сохранее
        сереализованный в json словарь.

    :param n: количество слов в устойчивой конструкции 
        По умолчанию = 3

    :param t: порог вхождения в словарь-
        для вхождения, конструкция должна встретиться t+1 раз и более
        По умолчанию = 1
    """
    if inp == '':
        ctx.exit_with_error('На вход не переданы файлы!', 1)
    OCCURRENCE_THRESHOLD=t
    
    paths=inp.split(',')

    wdict = process_frequent_wording_files(paths, n)
    if(len(wdict) == 0):
        print("Устойчивых конструкций, удовлетворяющих требованиям, не найдено.")
    else:
        print(wdict)

    if(outp != ''):
        serialize_dict(outp, wdict)


@app.command()
def recover(ctx: cli.Context, d: str = '', inp: str = ''):
    """
    Поиск и восстановление поврежденных слов в тексте с предварительно размеченными
    поврежденными местами, используя словарь устойчивых конструкций.
    В качестве символа, обозначающего поврежденные места в тексте используется 'ƪ'

    :param d: путь к файлу словаря устойчивых конструкций.

    :param inp: текст, подлежащий восстановлению.
    """
    wdict = deserialize_dict(d)
    if(len(wdict) == 0):
        ctx.exit_with_error('Словарь устойчивых конструкций не содержит в себе элементов', 2)

    N_NUMBER = len(list(wdict.keys())[0])
    print("Длина устойчивых конструкций словаря: {0}".format(N_NUMBER))

    output = search_in_dict(inp, wdict, N_NUMBER)
    if(len(output) == 0):
        print("Слов для восстановления не найдено")
    else:
        print(output)
#################################################################
#endregion