#!/usr/bin/env python
# coding: utf-8

# In[758]:


"""
Prediction 1:
scenario_1: The word test appears in at least in one field
scenario_2: All fields are at most the same
scenario_3: One word field
scenario_4: Random characters in at least in one field
"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

nltk.download("words")
nltk.download("stopwords")

# In[759]:


# import spam collection messages
column_names = ["label", "title"]
message_data = pd.read_table("SMSSpamCollection", names=column_names)

# In[760]:


# Remove rows that contain spam label and reset indexes
message_data = message_data.drop(
    message_data[message_data.label == "spam"].index
).reset_index(drop=True)

# In[761]:


# Drop label column
message_data = message_data.drop("label", axis=1)

# In[762]:


# Show a peek in the column
message_data.head(10)

# In[763]:


message_data.shape

# In[764]:


# load words into a dictionary for faster loading
english_words = dict.fromkeys(words.words(), None)


# In[765]:


def is_word(word):
    """
    Check if word is in english dictionary
    """
    try:
        english_words[word]
        return True
    except KeyError:
        return False


def get_false_word_indexes(data, percentage):
    """
    Check for words that are not words and return their index
    """
    indexes = []
    for index, message in enumerate(data["title"]):
        false_word_count = 0
        for word in message.split():
            if not is_word(word):
                false_word_count += 1
                if false_word_count / len(message) > percentage:
                    indexes.append(index)
                    break
    return indexes


def remove_single_words(dataframe):
    """
    Returns a single words from the dataframe
    """
    return [index for index, message in enumerate(dataframe["title"]) if len(message.split()) == 1]

# In[766]:


false_word_percentage = 0.02
indexes = get_false_word_indexes(message_data, false_word_percentage)
print("{} number of indexes will be deleted".format(len(indexes)))

# In[767]:


original_size = message_data.shape[0]
real_messages = message_data.drop(message_data.index[indexes]).reset_index(
    drop=True
)
print(
    "Data size went from {} to {}".format(
        original_size, real_messages.shape[0]
    )
)

# In[768]:


real_messages.head()

# In[769]:


indexes = remove_single_words(real_messages)
print("{} number of indexes will be deleted".format(len(indexes)))
no_single_messages = real_messages.drop(
    real_messages.index[indexes]
).reset_index(drop=True)
print(
    "Data size went from {} to {}".format(
        real_messages.shape[0], no_single_messages.shape[0]
    )
)


# In[770]:


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
    if prob < 0.45:
        choice = 0
    elif prob < 0.90:
        choice = len(split_words)
    else:
        choice = random.randint(0, len(split_words))
    return choice, split_words


# In[812]:


# Show that the test message in a random index of the sentence
add_test_message("Hello my name darryl")


# In[772]:


def shuffle_dataframe(df, num_shuffles):
    for x in range(num_shuffles):
        for column in df.columns:
            df[column] = df[column].sample(frac=1).reset_index(drop=True)
    return df


# In[773]:


stopwords_map = {word: 0 for word in stopwords.words("english")}
print
punctation_map = {word: 0 for word in string.punctuation}


def text_process(mess):
    nopunc = [char for char in mess if char not in punctation_map]
    nopunc = "".join(nopunc)
    return " ".join(
        [word for word in nopunc.split() if word.lower() not in stopwords_map]
    )


def create_default_scenario(df):
    scenario = pd.DataFrame(df)
    scenario["subtitle"] = scenario["title"]
    scenario["message"] = scenario["title"]
    for column in df.columns:
        scenario[column] = scenario[column].apply(text_process)
    return scenario


# In[774]:


def create_scenario_one(messages):
    df = create_default_scenario(messages)
    df = shuffle_dataframe(df, num_shuffles=30)
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
                df.loc[index][column] = add_test_message(df.loc[index][column])
    return df


# In[775]:


def create_scenario_two(messages):
    return create_default_scenario(messages)


# In[776]:


def create_scenario_three(messages):
    scenario = create_default_scenario(messages)
    for column in scenario.columns:
        scenario[column] = scenario[column].apply(lambda x: x.split()[0])
    return scenario


# In[777]:


def create_random_chars():
    num_chars = random.randint(5, 12)
    return "".join(
        [random.choice(string.ascii_letters) for _ in range(num_chars)]
    )


def create_scenario_four(messages):
    scenario = create_default_scenario(messages)
    for column in scenario.columns:
        scenario[column] = scenario[column].apply(
            lambda row: create_random_chars()
        )
    return scenario


# In[778]:


create_scenario_one(no_single_messages).to_csv("test_messages.csv")

# In[779]:


create_scenario_two(no_single_messages).head()

# In[780]:


create_scenario_three(no_single_messages).head()

# In[781]:


create_scenario_four(no_single_messages).head()

# In[782]:


# concatenate dataframes that has possible test messages
indexes = remove_single_words(real_messages)
no_single_messages = real_messages.drop(
    real_messages.index[indexes]
).reset_index(drop=True)
test_messages = pd.concat(
    [
        create_scenario_one(no_single_messages),
        create_scenario_two(no_single_messages),
        create_scenario_three(no_single_messages),
        create_scenario_four(no_single_messages),
    ]
)

# In[783]:


# add label column for distinguish that this is a test message
test_messages["label"] = 1
test_messages.head()

# In[784]:


indexes = remove_single_words(real_messages)
no_single_messages = real_messages.drop(
    real_messages.index[indexes]
).reset_index(drop=True)
no_test_messages = create_default_scenario(no_single_messages)
no_test_messages = shuffle_dataframe(no_test_messages, num_shuffles=50)

# In[785]:


no_test_messages["label"] = 0
no_test_messages.head()

# In[786]:


# concatenate dataframes test and not test messages
for x in range(50):
    df = (
        pd.concat([test_messages, no_test_messages])
        .sample(frac=1)
        .reset_index(drop=True)
    )

# In[787]:


df.head()

# In[788]:


demonstrate_df = pd.DataFrame(df)
demonstrate_df["title_length"] = demonstrate_df["title"].apply(
    lambda row: len(x.split())
)
demonstrate_df["subtitle_length"] = demonstrate_df["subtitle"].apply(
    lambda row: len(x.split())
)
demonstrate_df["message_length"] = demonstrate_df["message"].apply(
    lambda row: len(x.split())
)

# In[789]:


demonstrate_df.groupby("label").describe()

# In[790]:


demonstrate_df["title_length"].sort_values(ascending=False)[:5]

# In[791]:


demonstrate_df["title_length"].plot(bins=50, kind="hist")

# In[792]:


demonstrate_df["subtitle_length"].plot(bins=50, kind="hist")

# In[793]:


demonstrate_df["message_length"].plot(bins=50, kind="hist")

# In[794]:


# Cutoff was based on the graphs above
cutoff = 10
criteria1 = demonstrate_df["message_length"] < cutoff
criteria2 = demonstrate_df["subtitle_length"] < cutoff
criteria3 = demonstrate_df["title_length"] < cutoff
df = demonstrate_df[criteria1 & criteria2 & criteria3]


# In[795]:


def create_cvec_df(df, column, cvec):
    x = df[column]
    y = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=45
    )
    cvec.fit(x_train)
    train = pd.DataFrame(cvec.transform(x_train).todense())
    test = pd.DataFrame(cvec.transform(x_test).todense())
    return (
        {"train": train, "test": test, "ytrain": y_train, "ytest": y_test},
        cvec,
    )


# In[796]:


cvec = CountVectorizer(analyzer=text_process)
df = demonstrate_df.drop(
    ["message_length", "subtitle_length", "title_length"], axis=1
)
title_df, cvec = create_cvec_df(df, "title", cvec)
subtitle_df, cvec = create_cvec_df(df, "subtitle", cvec)
message_df, cvec = create_cvec_df(df, "message", cvec)
df.head(20)

# In[797]:


train = pd.concat(
    [title_df["train"], subtitle_df["train"], message_df["train"]], axis=1
)
test = pd.concat(
    [title_df["test"], subtitle_df["test"], message_df["test"]], axis=1
)

# In[798]:


# showcasing ytrain labels
print(title_df["ytrain"].head())
print(subtitle_df["ytrain"].head())
print(message_df["ytrain"].head())

# In[799]:


def show_results(clf, data, train, test):
    print("Classifier: {}\n".format(clf.fit(train, data["ytrain"])))
    predictions = clf.predict(test)
    show_wrong_classifictions(predictions, 20)
    print(classification_report(data["ytest"], predictions))
    print(confusion_matrix(data["ytest"], predictions))
    print("\n")
    print(
        "Average K-fold Accuracy score: {}".format(
            np.mean(cross_val_score(clf, train, data["ytrain"], cv=5))
        )
    )
    return clf


def show_wrong_classifictions(predictions, num_predictions):
    for index, prediction in enumerate(predictions[:num_predictions]):
        if prediction != message_df["ytest"].iloc[index]:
            print("Title: {}".format(demonstrate_df.loc[index]["title"]))
            print("Subtitle: {}".format(demonstrate_df.loc[index]["subtitle"]))
            print("Message: {}".format(demonstrate_df.loc[index]["message"]))
            print("Classifier's Prediction: {}".format(prediction))
            print(
                "Actual Prediction: {}".format(message_df["ytest"].iloc[index])
            )
            print("\n")


def show_sample_classifications(clf, cvec):
    title = ["W", "Hello", "test Synergy", "Footwear Sale", "test message"]
    subtitle = ["W", "Hello", "Synergy", "24 hour sale", "message"]
    message = [
        "W",
        "Hello",
        "test Synergy",
        "Buy one get one free",
        "test message",
    ]
    inputs = convert_fields(title, subtitle, message, cvec)
    predictions = clf.predict(inputs)
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            print(
                "Title: {} .. Subtitle: {} .. Message: {} ----> \
                Test message".format(
                    title[i], subtitle[i], message[i]
                )
            )
        else:
            print(
                "Title: {} .. Subtitle: {} .. Message: {} ----> \
                Not a test message".format(
                    title[i], subtitle[i], message[i]
                )
            )


def convert_fields(title, subtitle, message, cvec):
    title_cvec = pd.DataFrame(cvec.transform(title).todense())
    subtitle_cvec = pd.DataFrame(cvec.transform(subtitle).todense())
    message_cvec = pd.DataFrame(cvec.transform(message).todense())
    final_df = pd.concat([title_cvec, subtitle_cvec, message_cvec], axis=1)
    return final_df


# In[800]:


classifier = MultinomialNB()
show_results(classifier, message_df, train, test)
show_sample_classifications(classifier, cvec)

# In[801]:

svm_classifier = SVC()
show_results(svm_classifier, message_df, train, test)
show_sample_classifications(svm_classifier, cvec)
