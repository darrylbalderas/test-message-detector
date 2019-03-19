from unittest import TestCase
from preprocess import preprocess_text
import pandas as pd
import mock


def first_index_choice(split_words):
    return 0


def last_index_choice(split_words):
    return len(split_words)


def middle_index_choice(split_words):
    return len(split_words) // 2


class TestPreprocess(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "title": [
                    "A quick fox",
                    "fox",
                    "through the street",
                    "123456 1234",
                    "howdy",
                    "f;alsdj asldjas",
                ]
            }
        )
        self.words_df = pd.DataFrame(
            {"title": ["A quick fox", "fox", "through the street"]}
        )

        self.slight_false_words_df = pd.DataFrame(
            {
                "title": [
                    "A quick fox ate",
                    "fox ran through the forest",
                    "qws34 through the street",
                ]
            }
        )

    def test_remove_single_words_when_column_does_not_exist(self):
        self.assertEqual(
            preprocess_text.remove_single_words(self.df, "subtitle"), []
        )

    def test_remove_single_words(self):
        self.assertEqual(
            preprocess_text.remove_single_words(self.df, "title"), [1, 4]
        )

    def test_is_word(self):
        self.assertTrue(preprocess_text.is_word("hello"))
        self.assertTrue(preprocess_text.is_word("California"))
        self.assertFalse(preprocess_text.is_word("444"))
        self.assertFalse(preprocess_text.is_word("asdfghjk"))
        self.assertFalse(preprocess_text.is_word("strolling"))

    def test_get_false_word_indexes(self):
        self.assertEqual(
            preprocess_text.get_false_word_indexes(self.df, 0, "title"),
            [3, 5],
        )
        self.assertEqual(
            preprocess_text.get_false_word_indexes(
                self.words_df, 0, "title"
            ),
            [],
        )
        self.assertEqual(
            preprocess_text.get_false_word_indexes(
                self.slight_false_words_df, 0.25, "title"
            ),
            [2],
        )

        self.assertEqual(
            preprocess_text.get_false_word_indexes(
                self.df, 0.25, "subtitle"
            ),
            [],
        )

    @mock.patch(
        "preprocess.preprocess_text.get_index_choice",
        side_effect=first_index_choice,
    )
    def test_add_test_message_first_index(self, get_random_choice):
        self.assertEqual(
            preprocess_text.add_test_message("hello everyone"),
            "test hello everyone",
        )
        self.assertEqual(
            preprocess_text.add_test_message("goodmorning everyone"),
            "test goodmorning everyone",
        )

    @mock.patch(
        "preprocess.preprocess_text.get_index_choice",
        side_effect=last_index_choice,
    )
    def test_add_test_message_last_index(self, get_random_choice):
        self.assertEqual(
            preprocess_text.add_test_message("hello everyone"),
            "hello everyone test",
        )
        self.assertEqual(
            preprocess_text.add_test_message("goodmorning everyone"),
            "goodmorning everyone test",
        )

    @mock.patch(
        "preprocess.preprocess_text.get_index_choice",
        side_effect=middle_index_choice,
    )
    def test_add_test_message_last_index(self, get_random_choice):
        self.assertEqual(
            preprocess_text.add_test_message("hello everyone"),
            "hello test everyone",
        )
        self.assertEqual(
            preprocess_text.add_test_message("goodmorning everyone"),
            "goodmorning test everyone",
        )

    def test_remove_punctation(self):
        self.assertEqual(
            preprocess_text.remove_punctation("Hello!!!"), "Hello"
        )
        self.assertEqual(
            preprocess_text.remove_punctation("Stop123!!! There123"),
            "Stop There",
        )
        self.assertEqual(
            preprocess_text.remove_punctation("123!!! There123"), "There"
        )

    def test_remove_stopwords(self):
        self.assertEqual(
            preprocess_text.remove_stop_words("i love you"), "love"
        )
        self.assertEqual(
            preprocess_text.remove_stop_words(
                "We should go to that place"
            ),
            "go place",
        )
        self.assertEqual(
            preprocess_text.remove_stop_words("The person is the best"),
            "person best",
        )

    def test_apply_stemming(self):
        self.assertEqual(
            preprocess_text.apply_stemming("Lovely view"), "love view"
        )
        self.assertEqual(
            preprocess_text.apply_stemming("creating the view"),
            "creat the view",
        )

    def test_clean_sentence(self):
        self.assertEqual(
            preprocess_text.clean_sentence("123!!! Lower Lovely"),
            "lower lovely",
        )
        self.assertEqual(
            preprocess_text.clean_sentence("123!!! I Lower Lovely"),
            "lower lovely",
        )
        self.assertEqual(
            preprocess_text.clean_sentence("123!!! The Lower the Lovely"),
            "lower lovely",
        )

    def test_shuffle(self):
        self.assertNotEqual(
            list(preprocess_text.shuffle(self.df, 2)["title"]),
            list(self.df["title"]),
        )

    def test_remove_test(self):
        self.assertEqual(
            preprocess_text.remove_test_word("test word in the sentence"),
            "word in the sentence",
        )
        self.assertEqual(
            preprocess_text.remove_test_word("test test test"), ""
        )
        self.assertEqual(
            preprocess_text.remove_test_word("Test TEST teST"), ""
        )

    def test_remove_special_words(self):
        self.assertEqual(
            preprocess_text.remove_single_characters("ing n"), "ing"
        )
        self.assertEqual(
            preprocess_text.remove_single_characters("here ing n you"),
            "here ing  you",
        )
