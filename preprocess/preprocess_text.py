from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import random
import re
import pandas as pd

# nltk.download("words")
# nltk.download("stopwords")
english_words = dict.fromkeys(words.words(), None)
stopwords_map = {word: 0 for word in stopwords.words("english")}


def clean_sentence(sentence):
    sentence = remove_punctation(sentence)
    sentence = remove_stop_words(sentence)
    sentence = sentence.lower()
    return remove_test_word(sentence)


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
    except KeyError:
        return False
    return True


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
        index
        for index, sentence in enumerate(dataframe[column])
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


def remove_punctation(sentence):
    return " ".join(
        [re.sub(r"[^a-zA-Z]", "", word) for word in sentence.split()]
    ).strip()


def remove_stop_words(sentence):
    return " ".join(
        [
            word
            for word in sentence.split()
            if word.lower() not in stopwords_map
        ]
    )


def apply_stemming(sentence):
    return " ".join(
        [PorterStemmer().stem(word) for word in sentence.split()]
    )


def shuffle(df, num_shuffles):
    new_df = pd.DataFrame()
    for _ in range(num_shuffles):
        for column in df.columns:
            new_df[column] = (
                df[column].sample(frac=1).reset_index(drop=True)
            )
    return new_df


def create_default_scenario(df):
    scenario = df.copy()
    scenario["subtitle"] = scenario["title"]
    scenario["message"] = scenario["title"]
    for column in df.columns:
        scenario[column] = scenario[column].apply(clean_sentence)
    return scenario


def create_scenario_one(messages):
    df = create_default_scenario(messages)
    df = shuffle(df, num_shuffles=30)
    for index in df.index:
        prob = random.random()
        choice = random.choice(df.columns)
        if prob < 0.05:
            df.loc[index][choice] = add_test_message(df.loc[index][choice])
        elif prob < 0.15:
            columns = list(df.columns)
            df.loc[index][choice] = add_test_message(df.loc[index][choice])
            columns.remove(choice)
            choice = random.choice(columns)
            df.loc[index][choice] = add_test_message(df.loc[index][choice])
        else:
            for column in df.columns:
                df.loc[index][column] = add_test_message(
                    df.loc[index][column]
                )
    return df


def create_scenario_two(messages):
    return create_default_scenario(messages)


def create_scenario_three(messages):
    scenario = create_default_scenario(messages)
    for column in scenario.columns:
        scenario[column] = scenario[column].apply(lambda x: x.split()[0])
    return scenario


def create_scenario_four(messages):
    scenario = create_default_scenario(messages)
    for column in scenario.columns:
        scenario[column] = scenario[column].apply(
            lambda row: create_random_chars(5, 15)
        )
    return scenario


def create_scenario_five(messages):
    """
    Create fields that incorporate optional fields
    """
    df = create_default_scenario(messages)
    df = shuffle(df, num_shuffles=30)
    for index in df.index:
        prob = random.random()
        choice = random.choice(df.columns)
        if prob < 0.33:
            df.loc[index][choice] = ""
        elif prob < 0.66:
            for column in df.columns:
                if column == choice:
                    continue
                df.loc[index][column] = ""

        else:
            for column in df.columns:
                if column == "message":
                    continue
                df.loc[index][column] = ""
    return df


def create_scenario_six(messages):
    """
    Create fields that incorporate optional fields and at least one field
    has the word test
    """
    df = create_default_scenario(messages)
    df = shuffle(df, num_shuffles=30)
    for index in df.index:
        prob = random.random()
        choice = random.choice(df.columns)
        if prob < 0.33:
            for column in df.columns:
                if column == choice:
                    df.loc[index][column] = add_test_message(df.loc[index][column])
                else:
                    df.loc[index][column] = ""
        elif prob < 0.66:
            for column in df.columns:
                if column == choice:
                    df.loc[index][column] = ""
                else:
                    df.loc[index][column] = add_test_message(df.loc[index][column])
        else:
            for column in df.columns:
                if column == "message":
                    df.loc[index]["message"] = add_test_message(df.loc[index]["message"])
                else:
                    df.loc[index][column] = ""
    return df


def create_random_chars(low, high):
    letters = (
        "a b c d e f g h i j k l m n o p q r "
        "s t u v w x y z A B C D E F G H I J K L "
        "M N O P Q R S T U V W X Y Z".split(" ")
    )
    return "".join(
        [random.choice(letters) for _ in range(random.randint(low, high))]
    )


def remove_test_word(sentence):
    return " ".join(
        [re.sub(r"test", "", word.lower()) for word in sentence.split()]
    ).strip()


def remove_single_characters(sentence):
    return " ".join(
        [
            re.sub(r"\b[a-zA-Z]\b", "", word.lower())
            for word in sentence.split()
        ]
    ).strip()


def remove_long_sentences(sentence):
    if len(sentence.split()) > 10:
        return ""
    return sentence

