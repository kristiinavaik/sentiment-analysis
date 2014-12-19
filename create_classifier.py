try:
    import cPickle as pickle
    print("cPickle import successful")
except ImportError:
    print("failed to import cPickle, falling back to pickle")
    import pickle

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
 
def word_feats(words):
    return {word: True for word in words}

print("Finding ids for positive and negative reviews")
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

print("Creating feature sets")
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

print("Calculating cutoffs")
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)  # *3/4

print("Creating training set")
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
print("Creating test set")
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()

print("Serialize the classifier")
with open('cl.pkl', 'wb') as out:
    pickle.dump(classifier, out, pickle.HIGHEST_PROTOCOL)
