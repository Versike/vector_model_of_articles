import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()


# Обработка текста(Удаление стоп-слов, лемматизация)
def text_processing(main_text, all_sentences):
    english_stopwords = stopwords.words("english")
    sentences = nltk.sent_tokenize(main_text)
    for sentence in sentences:
        sentence = remove_exception_symbols(sentence)
        sentence = ' '.join([word for word in sentence.split() if len(word) > 2])
        process_sentence = remove_stopword_from_text(nltk.word_tokenize(sentence.lower()), english_stopwords)
        result_sentence = lemmatization(process_sentence)
        all_sentences.append(result_sentence)
    return all_sentences


# Удаление спец. Символов
def remove_exception_symbols(text):
    reg = re.compile('[^a-zA-Z- ]')
    text = reg.sub('', text)
    text = re.sub('-', ' ', text)
    return text


# Удаление стоп-слов из текста
def remove_stopword_from_text(text, stopwords):
    return [token for token in text if token not in stopwords]


# Лемматизация
def lemmatization(text):
    return [wnl.lemmatize(word) for word in text]


# Преобразование двумерного списка в одномерный
def transormation(all_words):
    result = []
    for word in all_words:
        result.extend(word if isinstance(word, list) else [word])
    return result


def create_list_all_sentences(main_text):
    reg_text = re.findall(r'(?:<TEXT>)\n(.+)\n (?:</TEXT>)', main_text)
    result_text = []
    for text in reg_text:
        text_processing(text, result_text)
    return result_text




