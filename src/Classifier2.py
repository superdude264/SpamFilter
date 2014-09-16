from collections import Counter
from Tag import Tag
import re

class Classifier2:

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
        self._tokenSpamicity = {}

    def train(self, msgs):
        # Build preliminary lexicon.
        for msg in msgs:
            tokens = self._tokenizeMessage(msg)
            if msg.tag == Tag.HAM:
                self._hamTokens += tokens
            elif msg.tag == Tag.SPAM:
                self._spamTokens += tokens

        for token, count in (self._hamTokens + self._spamTokens).items():
            if count > self._countThreshold:
                hamProb = self._tokenProbability(token, self._hamTokens)
                spamProb = self._tokenProbability(token, self._spamTokens)
                spamicity = min(.99999999, max(.01, spamProb / (hamProb + spamProb)))
                self._tokenSpamicity[token] = spamicity

    def _tokenProbability(self, word, klassCol):
        numOccurOfTokenInClass = klassCol[word]
        totalOccurOfAllTokensInClass = 0
        for _ , count in klassCol.items(): totalOccurOfAllTokensInClass += count
        totalUniqueTokensInClass = len(klassCol)
        prob = (float(numOccurOfTokenInClass) + self._smoothingFactor) / (totalOccurOfAllTokensInClass + self._smoothingFactor * totalUniqueTokensInClass)
        return prob

    def classify(self, msgs):
        for msg in msgs:
            tokenProbs = []
            for token, _ in self._tokenizeMessage(msg).items():
                tokenProb = self._tokenSpamicity.get(token, .5)
                tokenProbs.append(tokenProb)
            msgProb = self._combineProbabilities(tokenProbs)
            if msgProb >= .5: msg.spamProb = 1

    def _combineProbabilities(self, probList):
        product = 1.0
        inverseProduct = 1.0

        for prob in probList:
            product = product * prob
            inverseProduct = inverseProduct * (1.0 - prob)

        comboProb = product / (product + inverseProduct)
        return comboProb;

    def _tokenizeMessage(self, msg):
        msgText = msg.subject + " " + msg.body
        tokens = [w.strip() for w in re.split('\W', msgText) if w.isalpha()]
        return Counter(tokens)

