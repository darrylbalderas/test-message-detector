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
            preprocess_text.get_false_word_indexes(self.df, 0, "title"), [3, 5]
        )
        self.assertEqual(
            preprocess_text.get_false_word_indexes(self.words_df, 0, "title"), []
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

    @mock.patch('preprocess.preprocess_text.get_index_choice', side_effect=first_index_choice)
    def test_add_test_message_first_index(self, get_random_choice):
        self.assertEqual(preprocess_text.add_test_message("hello everyone"), "test hello everyone")
        self.assertEqual(preprocess_text.add_test_message("goodmorning everyone"), "test goodmorning everyone")

    @mock.patch('preprocess.preprocess_text.get_index_choice', side_effect=last_index_choice)
    def test_add_test_message_last_index(self, get_random_choice):
        self.assertEqual(preprocess_text.add_test_message("hello everyone"), "hello everyone test")
        self.assertEqual(preprocess_text.add_test_message("goodmorning everyone"), "goodmorning everyone test")

    @mock.patch('preprocess.preprocess_text.get_index_choice', side_effect=middle_index_choice)
    def test_add_test_message_last_index(self, get_random_choice):
        self.assertEqual(preprocess_text.add_test_message("hello everyone"), "hello test everyone")
        self.assertEqual(preprocess_text.add_test_message("goodmorning everyone"), "goodmorning test everyone")

