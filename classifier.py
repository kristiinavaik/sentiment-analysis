import os
import re
import string
import time

import nltk


CLASSIFIER_FILE = 'cl.pkl'


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
