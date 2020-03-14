from abc import ABC, abstractmethod
from icu import Locale, BreakIterator
from collections import namedtuple


RangeItem = namedtuple('RangeItem', [
    'x', 'y', 'text'
])


class TokenizerBase(ABC):
    def __init__(self, locale):
        self.locale = Locale(locale)
        self.breaker = self._get_breaker(self.locale)

    @abstractmethod
    def _get_breaker(self, locale):
        pass

    def get_ranges(self, s):
        self.breaker.setText(s)
        x = 0
        LRtn = []
        for y in list(self.breaker):
            LRtn.append((x, y))
            x = y
        return LRtn

    def get_segments(self, s):
        self.breaker.setText(s)
        x = 0
        LRtn = []
        for y in list(self.breaker):
            LRtn.append(s[x:y])
            x = y
        return LRtn

    def get_segments_as_range_items(self, s):
        self.breaker.setText(s)
        x = 0
        LRtn = []
        for y in list(self.breaker):
            LRtn.append(RangeItem(x, y, s[x:y]))
            x = y
        return LRtn


class SentenceTokenizer(TokenizerBase):
    def _get_breaker(self, locale):
        return BreakIterator.createSentenceInstance(locale)


class WordTokenizer(TokenizerBase):
    def _get_breaker(self, locale):
        return BreakIterator.createWordInstance(locale)


if __name__ == '__main__':
    print(SentenceTokenizer(locale='en').get_segments("test test TEST."))
    print(WordTokenizer(locale='en').get_segments("test test TEST."))
