import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def show_results(clf, x_train, x_test, y_train, y_test, df):
    clf = clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    # show_wrong_classifictions(predictions, y_test, df)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(
        "Average K-fold Accuracy score: {}".format(
            np.mean(cross_val_score(clf, x_train, y_train, cv=5))
        )
    )
    return clf


def split_train_test(df, column):
    x = df[column]
    y = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )
    return ({
        "xtrain": x_train,
        "xtest": x_test,
        "ytrain": y_train,
        "ytest": y_test,
    })


def show_wrong_classifictions(predictions, ytest, df):
    for i, index in enumerate(ytest.index):
        if predictions[i] != ytest.iloc[i]:
            print("Title: {}".format(df.loc[index]["title"]))
            print("Subtitle: {}".format(df.loc[index]["subtitle"]))
            print("Message: {}".format(df.loc[index]["message"]))
            print("Classifier's Prediction: {}".format(predictions[i]))
            print("Actual Prediction: {}".format(ytest.iloc[i]))
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


def update_train_test_df(cvec, title_df):
    title_df['xtrain'] = pd.DataFrame(cvec.transform(title_df['xtrain']).todense())
    title_df['xtest'] = pd.DataFrame(cvec.transform(title_df['xtest']).todense())


def main():
    engage_test = pd.read_csv("engageTest.csv")
    engage_test = engage_test.replace(np.nan, "")
    spam_test = pd.read_csv("spamTest.csv")
    spam_test = spam_test.replace(np.nan, "")
    df = pd.concat([engage_test, spam_test])
    for x in range(50):
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    title_df = split_train_test(df, "title")
    cvec = CountVectorizer()
    cvec = cvec.fit(title_df['xtrain'])
    update_train_test_df(cvec, title_df)

    subtitle_df = split_train_test(df, "subtitle")
    update_train_test_df(cvec, subtitle_df)

    message_df = split_train_test(df, "message")
    update_train_test_df(cvec, message_df)

    xtrain = pd.concat([title_df["xtrain"], subtitle_df["xtrain"], message_df["xtrain"]], axis=1)
    xtest = pd.concat([title_df["xtest"], subtitle_df["xtest"], message_df["xtest"]], axis=1)

    clf = DecisionTreeClassifier(random_state=42, max_depth=64)
    clf = show_results(clf, xtrain, xtest, title_df["ytrain"], title_df["ytest"], df)
    show_sample_classifications(clf, cvec)


if __name__ == "__main__":
    main()