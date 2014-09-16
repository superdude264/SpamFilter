from collections import Counter
from Tag import Tag
import re
import numpy as np

class Classifier:

    def __init__(self, priorSpam, countThreshold, smoothingFactor, defaultProb):
        self._hamTokens = Counter()
        self._spamTokens = Counter()
        self._hamHistogram = {}
        self._spamHistogram = {}
        self._priorSpam = priorSpam
        self._priorHam = 1 - priorSpam
        self._countThreshold = countThreshold
        self._smoothingFactor = smoothingFactor
        self._defaultProb = defaultProb

    def train(self, msgs):
        # Build preliminary lexicon.
        for msg in msgs:
            tokens = self._tokenizeMessage(msg)
            if msg.tag == Tag.HAM:
                self._hamTokens += tokens
            elif msg.tag == Tag.SPAM:
                self._spamTokens += tokens

        # Remove words whose counts aren't above the threshold.
        delList = []
        for token, count in (self._hamTokens + self._spamTokens).items():
            if count <= self._countThreshold:
                delList.append(token)
        self._hamTokens = Counter(dict([(word, count) for word, count in self._hamTokens.items() if word not in delList]))
        self._spamTokens = Counter(dict([(word, count) for word, count in self._spamTokens.items() if word not in delList]))
    
        # Calculate probabilities w/ smoothing.
        self._hamHistogram = dict([(word, self._tokenProbability(word, self._hamTokens)) for word, count in self._hamTokens.items()])
        self._spamHistogram = dict([(word, self._tokenProbability(word, self._spamTokens)) for word, count in self._hamTokens.items()])

    def _tokenProbability(self, word, klassCol):
        numOccurOfTokenInClass = klassCol[word]
        totalOccurOfAllTokensInClass = 0
        for _ , count in klassCol.items(): totalOccurOfAllTokensInClass += count
        totalUniqueTokensInClass = len(klassCol)
        prob = (float(numOccurOfTokenInClass) + self._smoothingFactor) / (totalOccurOfAllTokensInClass + self._smoothingFactor * totalUniqueTokensInClass)
        return prob

    def classify(self, msgs):
        for msg in msgs:
            hamProb = np.log2(self._priorHam)
            spamProb = np.log2(self._priorSpam)
            for word, _ in self._tokenizeMessage(msg).items():
                hamProb += self._hamHistogram.get(word, np.log2(self._defaultProb))
                spamProb += self._spamHistogram.get(word, np.log2(self._defaultProb))
            msg.hamProb = hamProb
            msg.spamProb = spamProb

    def _tokenizeMessage(self, msg):
        msgText = msg.subject + " " + msg.body
        tokens = [w.strip() for w in re.split('\W', msgText) if w.isalpha()]
        return Counter(tokens)

