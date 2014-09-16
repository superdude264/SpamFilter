import sys
import os
from Message import Message
from Tag import Tag
from Classifier import Classifier
from Classifier2 import Classifier2

# Paths
MSG_ROOT = "../emails/"
HAM_TRAINING_PATH = MSG_ROOT + "hamtraining/"
SPAM_TRAINING_PATH = MSG_ROOT + "spamtraining/"
HAM_TESTING_PATH = MSG_ROOT + "hamtesting/"
SPAM_TESTING_PATH = MSG_ROOT + "spamtesting/"

# Model Parameters
COUNT_THRESHOLD = 4
SMOOTHING_FACTOR = 3
DEFAULT_PROBABILITY = .5
SPAMINESS_THRESHOLD = 0

def main():
    try:
        trainingData, tuningData, testData, priorSpam = buildDataSets()
        nbc = Classifier(priorSpam, COUNT_THRESHOLD, SMOOTHING_FACTOR, DEFAULT_PROBABILITY)
        # nbc = Classifier2(priorSpam, 0, .01, None)
        nbc.train(trainingData)

        nbc.classify(testData)
        report(testData)

    except Exception as e:
        print e
        return 5

def report(dataSet):
    # Report
    correctHam = 0
    correctSpam = 0
    spamCount = 0
    for msg in dataSet:
        print msg
        if msg.tag == Tag.HAM and not msg.isSpam():
            correctHam += 1
        elif msg.tag == Tag.SPAM:
            spamCount += 1
            if msg.isSpam(): correctSpam += 1

    num = len(dataSet)
    numCorrect = correctHam + correctSpam
    numWrong = num - numCorrect
    pctCorrect = float(numCorrect) / num
    pctHamCorrect = float(correctHam) / (num - spamCount)
    pctSpamCorrect = float(correctSpam) / spamCount

    print
    print "Messages: %d" % num
    print "Correct: %d" % numCorrect
    print "Incorrect: %d" % numWrong
    print "Percent Correct: %f" % (pctCorrect * 100)
    print "Percent Ham Correct: %f" % (pctHamCorrect * 100)
    print "Percent Spam Correct: %f" % (pctSpamCorrect * 100)

def buildDataSets():
    # Build file lists.
    full_ham_training_files = os.listdir(HAM_TRAINING_PATH)
    ham_training_files = full_ham_training_files[:len(full_ham_training_files) / 2]
    ham_tuning_files = full_ham_training_files[len(full_ham_training_files) / 2:]

    # hack to increase training set at expense of tuning set.
    h2 = ham_tuning_files[:len(ham_tuning_files) / 2]
    ham_tuning_files = ham_tuning_files[len(ham_tuning_files) / 2:]
    ham_training_files += h2

    full_spam_training_files = os.listdir(SPAM_TRAINING_PATH)
    spam_training_files = full_spam_training_files[:len(full_spam_training_files) / 2]
    spam_tuning_files = full_spam_training_files[len(full_spam_training_files) / 2:]

    # hack to increase training set at expense of tuning set.
    s2 = spam_tuning_files[:len(spam_tuning_files) / 2]
    spam_tuning_files = spam_tuning_files[len(spam_tuning_files) / 2:]
    spam_training_files += s2

    # Build data sets.
    training_data = Message.createBulk(HAM_TRAINING_PATH, ham_training_files) + Message.createBulk(SPAM_TRAINING_PATH, spam_training_files)
    tuning_data = Message.createBulk(HAM_TRAINING_PATH, ham_tuning_files) + Message.createBulk(SPAM_TRAINING_PATH, spam_tuning_files)
    test_data = Message.createBulk(HAM_TESTING_PATH, os.listdir(HAM_TESTING_PATH)) + Message.createBulk(SPAM_TESTING_PATH, os.listdir(SPAM_TESTING_PATH))

    # Calculate prior spam probability (% of training set that is spam).
    priorSpam = float(len(full_spam_training_files)) / (len(full_ham_training_files) + len(full_spam_training_files))

    return (training_data, tuning_data, test_data, priorSpam)

if __name__ == "__main__":
    sys.exit(main())
