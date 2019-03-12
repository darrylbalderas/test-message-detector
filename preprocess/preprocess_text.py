from nltk.corpus import words
from nltk.corpus import stopwords
import nltk
import random

nltk.download("words")
nltk.download("stopwords")
english_words = dict.fromkeys(words.words(), None)


def remove_single_words(dataframe, column):
    if column not in dataframe.columns:
        return []
    return [
        index
        for index, message in enumerate(dataframe[column])
        if len(message.split()) == 1
    ]


def is_word(word):
    try:
        english_words[word]
        return True
    except KeyError:
        return False


def above_percentage_threshold(sentence, percentage):
    false_word_count = 0
    split_words = sentence.split()
    for word in split_words:
        if not is_word(word):
            false_word_count += 1
            if false_word_count / len(split_words) >= percentage:
                return True
    return False


def get_false_word_indexes(dataframe, percentage, column):
    if column not in dataframe.columns:
        return []
    return [
        i
        for i, sentence in enumerate(dataframe["title"])
        if above_percentage_threshold(sentence, percentage)
    ]


def add_test_message(sentence):
    split_words = sentence.split()
    choice = get_index_choice(split_words)
    update_sentence = (
        " ".join(split_words[:choice])
        + " test "
        + " ".join(split_words[choice:])
    )
    return update_sentence.strip()


def get_index_choice(split_words):
    prob = random.random()
    if prob < 0.40:
        choice = 0
    elif prob < 0.80:
        choice = len(split_words)
    else:
        choice = random.randint(0, len(split_words))
    return choice
