import pandas as pd
import preprocess_text as pt


def create_spam_train_data():
    column_names = ["label", "title"]
    data = pd.read_table("SMSSpamCollection", names=column_names)
    data = data[data["label"] != "spam"].reset_index(drop=True)
    data = data.drop("label", axis=1)
    create_train_data(data, "spamTest.csv")


def create_engage_train_data():
    data = pd.read_csv("contentInfo.csv")
    data.columns = ["title"]
    create_train_data(data, "engageTest.csv")


def create_train_data(data, output_file):
    data["title"] = data["title"].apply(pt.clean_sentence)
    data["title"] = data["title"].apply(pt.remove_single_characters)
    data["title"] = data["title"].apply(pt.remove_long_sentences)
    indexes = pt.remove_single_words(data, "title")
    data = data.drop(indexes, axis=0).reset_index(drop=True)
    false_word_percentage = 0
    indexes = pt.get_false_word_indexes(
        data, false_word_percentage, "title"
    )
    data = data.drop(indexes, axis=0).reset_index(drop=True)
    data["title"] = data[data["title"] != ""].reset_index(drop=True)
    data = data.dropna()
    test_messages = pd.concat(
        [
            pt.create_scenario_one(data),
            pt.create_scenario_two(data),
            pt.create_scenario_three(data),
            pt.create_scenario_four(data),
            pt.create_scenario_six(data)
        ]
    )
    test_messages["label"] = 1
    no_test_messages = pd.concat([pt.create_default_scenario(data), pt.create_scenario_five(data)])
    number_shuffles = 50
    no_test_messages = pt.shuffle(
        no_test_messages, num_shuffles=number_shuffles
    )
    no_test_messages["label"] = 0
    output_df = pd.concat([test_messages, no_test_messages])
    for x in range(number_shuffles):
        output_df = output_df.sample(frac=1).reset_index(drop=True)
    output_df.to_csv(output_file, index=False)


def main():
    create_engage_train_data()
    create_spam_train_data()


if __name__ == "__main__":
    main()
