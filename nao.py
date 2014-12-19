# -*- encoding: UTF-8 -*-

import sys
import time

from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule

from optparse import OptionParser

from classifier import Classifier

if sys.version_info.major == 2:
    input = raw_input

# NAO_IP = "nao.local"
NAO_IP = "localhost"
NAO_PORT = 9559


class TextAnalyzerModule(ALModule):

    def __init__(self, name):
        ALModule.__init__(self, name)
        self.classifier = Classifier()
        self.tts = ALProxy("ALTextToSpeech")

    def say(self, text):
        sentiment = self.classifier.classify(text)
        self.tts.say('That is %s!' % sentiment.get_string_value())

def main():
    myBroker = ALBroker("myBroker",
       "0.0.0.0",   # listen to anyone
       0,           # find a free port and use it
       NAO_IP,      # parent broker IP
       NAO_PORT     # parent broker port
   )

    text_analyzer = TextAnalyzerModule("TextAnalyzer")
    try:
        while True:
            text_analyzer.say(input("Say something to Nao: "))
    except KeyboardInterrupt:
        print
        print "Interrupted by user, shutting down"
        myBroker.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()
