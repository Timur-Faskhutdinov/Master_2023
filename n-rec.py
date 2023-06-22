__author__ = "Faskhutdinov Timur"
#region Imports
#################################################################
# Textblob
from textblob import TextBlob
from textblob import Word
from textblob import WordList
import nltk
nltk.download('punkt', quiet=True)

from collections import Counter
import re
import json

# Расстояние Левенштейна
from Levenshtein import distance

# Обработка pdf-файлов
import PyPDF2 as pdf2

# import cli
import click
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
@click.group()
def mainn():
    pass

@mainn.command()
@click.argument('input-files', nargs=-1, type=click.Path())
@click.option('--output', '-o', help='Файл, куда будет записан сереализованный в json словарь.', type=click.Path())
@click.option('--n-length', '-n', default=N_NUMBER, help=('Количество слов в устойчивой конструкции. По умолчанию = {0}'.format(N_NUMBER)))
@click.option('--treshold', '-t', default=OCCURRENCE_THRESHOLD, help=('Для вхождения в словарь, конструкция должна встретиться t+1 раз и более. По умолчанию = {0}'.format(OCCURRENCE_THRESHOLD)))
@click.pass_context
def generate(ctx, input_files, output, n_length, treshold):
    """
    Генерация словаря устойчивых конструкций. \n
    INPUT_FILES - Файлы текстов-источников, перечисленные через пробел.
    """
    OCCURRENCE_THRESHOLD=treshold
    N_NUMBER=n_length
    
    # paths=input_files.split(',')

    wdict = process_frequent_wording_files(input_files, n_length)
    if(len(wdict) == 0):
        click.echo("Устойчивых конструкций, удовлетворяющих требованиям, не найдено.")
    else:
        click.echo(wdict)
    
    if(output != None):
        serialize_dict(output, wdict)


@mainn.command()
@click.argument('dictionary')
@click.option('--input', '-i', prompt='Введите текст для восстановления, для поврежденных мест используйте символ {0}'.format(SUBSTITUTE_SYMBOL), help='Tекст, подлежащий восстановлению.')
@click.pass_context
def recover(ctx, dictionary, input):
    """
    Поиск и восстановление поврежденных слов в тексте с предварительно размеченными
    поврежденными местами, используя словарь устойчивых конструкций.
    В качестве символа, обозначающего поврежденные места в тексте используется 'ƪ'. \n
    DICTIONARY - файл, содержащий сереализованный словарь устойчивых конструкций.
    """
    wdict = deserialize_dict(dictionary)
    if(len(wdict) == 0):
        ctx.exit_with_error('Словарь устойчивых конструкций не содержит в себе элементов', 2)

    N_NUMBER = len(list(wdict.keys())[0])
    click.echo("Длина устойчивых конструкций словаря: {0}".format(N_NUMBER))

    output = search_in_dict(input, wdict, N_NUMBER)
    if(len(output) == 0):
        click.echo("Слов для восстановления не найдено")
    else:
        click.echo(output)

if __name__ == '__main__':
    mainn()
#################################################################
#endregion