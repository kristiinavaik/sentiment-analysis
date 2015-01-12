try:
    import cPickle as pickle
    print("cPickle import successful")
except ImportError:
    print("failed to import cPickle, falling back to pickle")
    import pickle
import os

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews as reviews
from nltk import word_tokenize


def word_feats(words):
    return {word: True for word in words}


def get_combined_features():
    with open(os.path.join('training_data', 'combined.txt')) as f:
        sentences = []
        for line in f:
            sentiment, sentence = line.split('\t')
            tokens = word_tokenize(sentence)
            sentences.append((word_feats(tokens), sentiment[:3]))
    return sentences

print("Finding ids for positive and negative reviews")
negids = reviews.fileids('neg')
posids = reviews.fileids('pos')

print("Creating feature sets")
negfeats = [(word_feats(reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(reviews.words(fileids=[f])), 'pos') for f in posids]
mixfeats = get_combined_features()

print("Calculating cutoffs")
negcutoff = int(len(negfeats)*9/10)
poscutoff = int(len(posfeats)*9/10)
mixcutoff = int(len(mixfeats)*9/10)

print("Creating training set")
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] + mixfeats[:mixcutoff]
print("Creating test set")
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] + mixfeats[mixcutoff:]
print('Train on %d instances, test on %d instances' %
      (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('Accuracy: %f' % nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

print("Serialize the classifier")
with open('cl.pkl', 'wb') as out:
    pickle.dump(classifier, out, pickle.HIGHEST_PROTOCOL)
