"""
TODO: Use https://fasttext.cc/docs/en/aligned-vectors.html

* Find definitions
* Find words in parallel sentences
* Perhaps could also allow training my own models?
* Output relative word probabilities
"""
import numpy as np
from collections import namedtuple

WordEmbeddings = namedtuple('WordEmbeddings', [
    'DFreqs', 'DFreqsToWord', 'LVectors'
])


def get_D_word_embeddings(path):
    DWords = {}
    DFreqs = {}
    DFreqsToWord = {}

    with open(path, 'r', encoding='utf-8') as f:
        for x, line in enumerate(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            DWords[word] = coefs
            DFreqs[word] = x  # More common words might be higher up(?)
            DFreqsToWord[x] = word

            if x % 1000 == 0:
                print(x)

            #if x > 20000:
            #    break

    a = np.ndarray(
        shape=(
            len(DWords),
            len(DWords[word])
        ),
        dtype='float32'
    )
    for word, vec in DWords.items():
        # Note the frequency is the actual index
        a[DFreqs[word], :] = vec
    return WordEmbeddings(DFreqs, DFreqsToWord, a)


ENGLISH = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.en.align.vec'
)
CHINESE = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.zh.align.vec'
)
INDONESIAN = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.id.align.vec'
)
GERMAN = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.de.align.vec'
)


def get_L_cands(word_embeddings, find_me):
    LCands = word_embeddings.LVectors - find_me
    LCands = np.sum(np.abs(LCands), axis=1)
    LSmallestIdx = np.argpartition(LCands, 30)[:30]  # ???

    LRtn = []
    for idx in LSmallestIdx:
        # (score, frequency/ID)
        LRtn.append((LCands[idx], word_embeddings.DFreqsToWord[idx]))
    LRtn.sort()
    return LRtn


def print_translations(word_embeddings, s):
    find_me = ENGLISH.LVectors[ENGLISH.DFreqs[s]]
    eng_freq = ENGLISH.DFreqs[s]
    LCands = get_L_cands(word_embeddings, find_me)

    LLikely = []
    LUnlikely = []

    def do_print(L):
        for score, cand in L:
            print(
                score, cand,
                f"English Freq: "
                f"{ENGLISH.DFreqs[s]}/{len(ENGLISH.DFreqs)}",
                f"Other Freq: "
                f"{word_embeddings.DFreqs[cand]}/{len(word_embeddings.DFreqs)}"
            )

    # Really uncommon words are
    # lower quality and likely junk
    for score, cand in LCands:
        freq = word_embeddings.DFreqs[cand]

        if freq > min(eng_freq, 2000)**1.3:
            # It should be pretty unlikely that the frequency in English
            # is an order of magnitude higher in the other language

            # (OPEN ISSUE: Perhaps it could be argued, very common words
            #  in one language may not be the best candidate for
            #  uncommon words?)
            LUnlikely.append((score, cand))
        else:
            LLikely.append((score, cand))

    print("==Likely candidates==")
    do_print(LLikely)
    print("==Unlikely candidates==")
    do_print(LUnlikely)


if __name__ == '__main__':
    #print_translations('cat')
    #print_translations('king')
    #print_translations('queen')

    while True:
        find_me = input("Enter a word to find:")
        try:
            print_translations(CHINESE, find_me)
            print_translations(INDONESIAN, find_me)
            print_translations(GERMAN, find_me)
        except KeyError:
            print(f"{find_me} was not found")
