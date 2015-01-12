# -*- encoding: UTF-8 -*-

import sys

from classifier import Classifier, Responder

if sys.version_info.major == 2:
    input = raw_input


def main():
    classifier = Classifier()
    responder = Responder()

    def say(text):
        sentiment = classifier.classify(text)
        response = responder.get_response(sentiment)
        print("Response: %s" % response)
    try:
        while True:
            say(input("Say something to Nao: "))
    except (EOFError, KeyboardInterrupt):
        print
        print "Interrupted by user, shutting down"
        sys.exit(0)

if __name__ == '__main__':
    main()
