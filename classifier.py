import os
import re
from random import randrange
import string
import time

import nltk


CLASSIFIER_FILE = 'cl.pkl'


class Responder(object):

    POSITIVE_RESPONSES_FILE = 'positive_responses.txt'
    NEGATIVE_RESPONSES_FILE = 'negative_responses.txt'
    NEUTRAL_RESPONSES_FILE = 'neutral_responses.txt'

    def __init__(self):
        self.positive_responses = self._init_responses(self.POSITIVE_RESPONSES_FILE)
        self.negative_responses = self._init_responses(self.NEGATIVE_RESPONSES_FILE)
        self.neutral_responses = self._init_responses(self.NEUTRAL_RESPONSES_FILE)
        self.used_positive_responses = []
        self.used_negative_responses = []
        self.used_neutral_responses = []

    def _init_responses(self, filename):
        responses = []
        with open(filename) as f:
            for line in f:
                if line.strip():
                    responses.append(line.strip())
        return responses

    def get_response(self, sentiment):
        if sentiment.is_positive():
            return self._get_positive_response()
        elif sentiment.is_negative():
            return self._get_negative_response()
        elif sentiment.is_neutral():
            return self._get_neutral_response()
        else:
            raise ValueError("Invalid sentiment")

    def _get_positive_response(self):
        if not self.positive_responses:
            self.positive_responses = self.used_positive_responses
            self.used_positive_responses = []
        index = randrange(0, len(self.positive_responses))
        response = self.positive_responses.pop(index)
        self.used_positive_responses.append(response)
        return response

    def _get_negative_response(self):
        if not self.negative_responses:
            self.negative_responses = self.used_negative_responses
            self.used_negative_responses = []
        index = randrange(0, len(self.negative_responses))
        response = self.negative_responses.pop(index)
        self.used_negative_responses.append(response)
        return response

    def _get_neutral_response(self):
        if not self.neutral_responses:
            self.neutral_responses = self.used_neutral_responses
            self.used_neutral_responses = []
        index = randrange(0, len(self.neutral_responses))
        response = self.neutral_responses.pop(index)
        self.used_neutral_responses.append(response)
        return response


class Sentiment(object):

    POSITIVE = 1
    NEGATIVE = 2
    NEUTRAL = 3

    POSITIVE_STR = 'positive'
    NEGATIVE_STR = 'negative'
    NEUTRAL_STR = 'neutral'

    def __init__(self, text, results):
        self.text = text
        self.results = results
        self.value = self._analyze_results(results)
        print(self)

    def is_positive(self):
        return self.value == self.POSITIVE

    def is_negative(self):
        return self.value == self.NEGATIVE

    def is_neutral(self):
        return self.value == self.NEUTRAL

    def has_same_value(self, other):
        """ Checks whether given sentiment has the same value """
        return self.value == other.value

    @classmethod
    def _analyze_results(cls, results):
        neg_count = results.count('neg')
        pos_count = len(results) - neg_count
        if neg_count > pos_count:
            return cls.NEGATIVE
        elif neg_count < pos_count:
            return cls.POSITIVE
        return cls.NEUTRAL

    def get_numberical_value(self):
        """ Returns either 1, 2 or 3 """
        return self.value

    def get_string_value(self):
        """ Returns either 'positive', 'negative' or 'neutral' """
        return {
            self.NEUTRAL: self.NEUTRAL_STR,
            self.POSITIVE: self.POSITIVE_STR,
            self.NEGATIVE: self.NEGATIVE_STR
        }.get(self.value, self.NEUTRAL_STR)

    def __str__(self):
        """ To string for Sentiment object """
        value = self.get_string_value()
        return '%s\n%s %s' % (self.text, value, self.results)


class Classifier(object):

    def __init__(self):
        self.translations = string.maketrans("", "")
        self.punctuation = string.punctuation.replace("'", "")
        self._init_classifier()

    def _init_classifier(self):
        """ Reads nltk classifier object from pickle """

        print("Initializing classifier")
        try:
            import cPickle as pickle
            print("cPickle import successful")
        except ImportError:
            print("failed to import cPickle, falling back to pickle")
            import pickle
        start = time.time()
        with open(CLASSIFIER_FILE, 'rb') as fid:
            self._classifier = pickle.load(fid)
        print("Created classifier in %.2f sec" % (time.time() - start))

    def _clear(self, word):
        """ Removes puncuation from given word and returns it lowercased """
        return word.lower().translate(self.translations, self.punctuation)

    def word_features(self, words):
        """ Compose a feature set of given list of words for nltk classifier """
        cleared = [self._clear(word) for word in words.split()]
        return {w: True for w in cleared if len(w) > 2}

    def classify(self, text):
        """ Returns a Sentiment object corresponding to the given text """
        sentences = re.split(r"[\.\?!]", text)
        feature_sets = [self.word_features(sentence) for sentence in sentences if sentence]
        results = self._classifier.classify_many(feature_sets)
        return Sentiment(text, results)
