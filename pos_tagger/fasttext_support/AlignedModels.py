"""
TODO: Use https://fasttext.cc/docs/en/aligned-vectors.html

* Find definitions
* Find words in parallel sentences
* Perhaps could also allow training my own models?
* Output relative word probabilities
"""
from collections import namedtuple
from sys import maxsize
import numpy as np

# Putting this here so I don't forget
# to enable the entire dictionaries
TEST_MODE = True


class AlignedVectors:
    def __init__(self, path):
        self.iso = path.split('wiki.')[1].split('.')[0]
        self.DFreqs, self.DFreqsToWord, self.LVectors = \
            self.__get_D_word_embeddings(path)

    #====================================================================#
    #                             Load Data                              #
    #====================================================================#

    def __get_D_word_embeddings(self, path):
        DWords = {}  # Note this isn't stored in `self`
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

                if x > 20000 and TEST_MODE:
                    break

        a = np.ndarray(
            shape=(
                len(DWords),
                len(DWords[word])
            ),
            dtype='float32'
        )
        for word, vec in DWords.items():
            # NOTE: the index is also the frequency of the word
            #   (with 0 being the most common), as the files
            #   are sorted in order of most to least common
            a[DFreqs[word], :] = vec
        return DFreqs, DFreqsToWord, a

    #====================================================================#
    #                              Getters                               #
    #====================================================================#

    def __iter__(self):
        for x, vec in enumerate(self.LVectors):
            yield x, self.DFreqsToWord[x], vec

    def word_index_to_word(self, word_index):
        """
        
        """
        return self.DFreqsToWord[word_index]

    def word_to_word_index(self, word):
        return self.DFreqs[word]

    def get_vector_for_word(self, word):
        return self.LVectors[
            self.word_to_word_index(word)
        ]

    #====================================================================#
    #                  Find Translations/Similar Words                   #
    #====================================================================#

    def get_similar_words(self, find_me, n=30):
        """
        Given a input vector for a given word,
        get the most similar words.

        :param find_me: a vector found using get_vector_for_word()
        """
        LCands = self.LVectors - find_me
        LCands = np.sum(np.abs(LCands), axis=1)
        LSmallestIdx = np.argpartition(LCands, n)[:n]

        LRtn = []
        for idx in LSmallestIdx:
            # (score, word_index/frequency)
            LRtn.append((LCands[idx], self.word_index_to_word(idx)))
        LRtn.sort()
        return LRtn

    def get_translations(self, other_aligned_vectors_inst, s):
        find_me = self.get_vector_for_word(s)
        LCands = other_aligned_vectors_inst.get_similar_words(find_me)
        return LCands

    def print_translations(self, other_aligned_vectors_inst, s):
        LCands = self.get_translations(
            other_aligned_vectors_inst, s
        )
        LLikely = []
        LUnlikely = []

        eng_freq = self.DFreqs[s]

        def do_print(L):
            for score, cand in L:
                print(
                    score, cand,
                    f"English Freq: "
                    f"{self.DFreqs[s]}/{len(self.DFreqs)}",
                    f"Other Freq: "
                    f"{other_aligned_vectors_inst.DFreqs[cand]}/"
                    f"{len(other_aligned_vectors_inst.DFreqs)}"
                )

        # Really uncommon words are
        # lower quality and likely junk
        for score, cand in LCands:
            freq = other_aligned_vectors_inst.word_to_word_index(cand)

            if freq > min(eng_freq, 2000) * 6:
                # It should be pretty unlikely that the word_index in English
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


#====================================================================#
#                        Sentence Alignment                          #
#====================================================================#


AlignedItem = namedtuple('AlignedItem', [
    'source_index', 'target_index',
    'source_text', 'target_text',
    'score'
])


def get_tokens(iso, s):
    # FOR TESTING ONLY!!! =========================================================
    return s.split()


def align_sentences(from_inst, to_inst,
                    from_s, to_s):

    # Tokenize the two sentences (if they haven't been already)
    # If using a POS system which tokenizes separately, it might
    # make sense to just use its output, and not do it again
    if isinstance(from_s, (list, tuple)):
        LFromTokens = from_s
    else:
        LFromTokens = get_tokens(from_inst.iso, from_s)

    if isinstance(to_s, (list, tuple)):
        LToTokens = to_s
    else:
        LToTokens = get_tokens(to_inst.iso, to_s)

    # Get the vectors for all words we can
    LFromVecs = []
    LToVecs = []

    for x, from_token in enumerate(LFromTokens):
        try:
            from_vec = from_inst.get_vector_for_word(
                from_token.lower()
            )
        except KeyError:
            from_vec = None
        LFromVecs.append(from_vec)

    for y, to_token in enumerate(LToTokens):
        try:
            to_vec = to_inst.get_vector_for_word(
                to_token.lower()
            )
        except KeyError:
            to_vec = None
        LToVecs.append(to_vec)

    # Get differential scores for all combinations
    a = np.full(
        shape=(len(LFromTokens), len(LToTokens)),
        dtype='float32',
        fill_value=maxsize
    )
    for x, from_token in enumerate(LFromTokens):
        for y, to_token in enumerate(LToTokens):
            from_vec = LFromVecs[x]
            to_vec = LToVecs[y]

            if from_vec is None or to_vec is None:
                continue  # Already filled with `maxsize`
            else:
                diff = np.sum(np.abs(from_vec-to_vec))
                a[x, y] = diff

    # Get a from/to map of indices in LFromTokens/LToTokens
    DFromToMap = {}
    DToFromMap = {}

    while len(DFromToMap) != min(len(LToVecs), len(LFromVecs)):
        idx1, idx2 = smallest_indices(
            a, min(len(LFromTokens), len(LToTokens))
        )
        #print("IDX:", idx1, idx2)

        for x, y in zip(idx1, idx2):
            if not x in DFromToMap and not y in DToFromMap:
                DFromToMap[x] = y
                DToFromMap[y] = x
            a[x, y] = maxsize

    # Output aligned tokens (AlignedItem's)
    # Won't output whitespace for now
    LFromRtn = []
    for x, from_token in enumerate(LFromTokens):
        if x in DFromToMap:
            LFromRtn.append(AlignedItem(x, DFromToMap[x],
                                        from_token, LToTokens[DFromToMap[x]],
                                        a[x, y]))
        else:
            LFromRtn.append(AlignedItem(x, None,
                                        from_token, None,
                                        None))

    LToRtn = []
    for y, to_token in enumerate(LToTokens):
        if y in DToFromMap:
            LToRtn.append(AlignedItem(y, DToFromMap[y],
                                      to_token, LFromTokens[DToFromMap[y]],
                                      a[x, y]))
        else:
            LToRtn.append(AlignedItem(y, None,
                                      to_token, None,
                                      None))
    return LFromRtn, LToRtn


def smallest_indices(ary, n):
    """
    Returns the n smallest indices from a numpy array.
    """
    flat = ary.flatten()
    indices = np.argpartition(flat, n)[:n] # CHECK ME!!
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, ary.shape)


if __name__ == '__main__':
    BASE_PATH = '/mnt/docs/dev/data/fast_text/aligned_word_vectors'
    ENGLISH = AlignedVectors(f'{BASE_PATH}/wiki.en.align.vec')

    print(
        align_sentences(ENGLISH, ENGLISH,
                        'I eat cake for breakfast nope',
                        'Cake I eat breakfast for')
    )

    CHINESE = AlignedVectors(f'{BASE_PATH}/wiki.zh.align.vec')

    print(align_sentences(
        ENGLISH, CHINESE,
        'May I eat that cake ?',
        '我 可以 吃 那个 蛋糕 吗 ？'
    ))

    INDONESIAN = AlignedVectors(f'{BASE_PATH}/wiki.id.align.vec')
    GERMAN = AlignedVectors(f'{BASE_PATH}/wiki.de.align.vec')

    while True:
        find_me = input("Enter a word to find:")
        try:
            ENGLISH.print_translations(CHINESE, find_me)
            ENGLISH.print_translations(INDONESIAN, find_me)
            ENGLISH.print_translations(GERMAN, find_me)
        except KeyError:
            print(f"{find_me} was not found")
