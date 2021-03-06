"""
TODO: Use https://fasttext.cc/docs/en/aligned-vectors.html

* Find definitions
* Find words in parallel sentences
* Perhaps could also allow training my own models?
* Output relative word probabilities
"""
import pickle
from os.path import exists

try:
    import cupy as np
    from cupy import linalg
    print("AlignedVectors: using cupy on GPU")
    using_gpu = True
except ImportError:
    import numpy as np
    from numpy import linalg
    print("AlignedVectors: using numpy on CPU")
    using_gpu = False

# Putting this here so I don't forget
# to enable the entire dictionaries
EMBEDDINGS_LIMIT = 100000

# TODO: Don't use this hardcoded path!
BASE_PATH = '/mnt/docs/dev/data/fast_text/aligned_word_vectors'

# Low frequency words actually yield remarkably low quality results!
CUTOFF_FREQUENCY = 700


class AlignedVectors:
    def __init__(self, path):
        self.iso = path.split('wiki.')[1].split('.')[0]
        self.DFreqs, self.DFreqsToWord, self.LVectors = \
            self.__get_D_word_embeddings(path)

    #====================================================================#
    #                             Load Data                              #
    #====================================================================#

    def __get_D_word_embeddings(self, path):
        if exists(f'{path}.{EMBEDDINGS_LIMIT}.npy'):
            with open(f'{path}.{EMBEDDINGS_LIMIT}.pkl', mode='rb') as f:
                DFreqs, DFreqsToWord = pickle.load(f)
            with open(f'{path}.{EMBEDDINGS_LIMIT}.npy', mode='rb') as f:
                a = np.load(f)
            return DFreqs, DFreqsToWord, a

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

                if x > EMBEDDINGS_LIMIT:
                    # A lot of the time it might be better to
                    # clip results - millions of results might actually reduce
                    # the quality as frequencies get lower!
                    break
                #print(word)

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

        with open(f'{path}.{EMBEDDINGS_LIMIT}.pkl', mode='wb') as f:
            pickle.dump((DFreqs, DFreqsToWord), f)
        with open(f'{path}.{EMBEDDINGS_LIMIT}.npy', mode='wb') as f:
            np.save(f, a)
        return DFreqs, DFreqsToWord, a

    #====================================================================#
    #                              Getters                               #
    #====================================================================#

    def __len__(self):
        return len(self.DFreqs)

    def iter(self, exclude_high_freq=True):
        for x, vec in enumerate(self.LVectors):
            if x < CUTOFF_FREQUENCY and exclude_high_freq:
                continue
            yield x, self.DFreqsToWord[x], vec

    def word_index_to_word(self, word_index):
        return self.DFreqsToWord[int(word_index)]

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

        # I've tried to make sure I'm not transferring many times
        # between gpu/main memory, as that can be very slow!!

        # Use cosine similarity
        # Could use sklearn, but best to use generic
        # numpy ops so as to be able to parallelize
        #from sklearn.metrics.pairwise import cosine_similarity
        #LCands = cosine_similarity(find_me.reshape(1, -1), self.LVectors).reshape(-1)

        a = find_me
        b = self.LVectors
        LCands = np.sum(a*b, axis=1)  # dot product for each row
        LCands = LCands / (linalg.norm(a) * linalg.norm(b, axis=1))
        LCands = LCands.reshape(-1)

        LLargestIdx = np.argpartition(LCands, -n)[-n:]
        LCands = LCands[LLargestIdx]

        if using_gpu:
            LLargestIdx = np.asnumpy(LLargestIdx)
            LCands = np.asnumpy(LCands)

        LRtn = []
        for idx, score in zip(LLargestIdx, LCands):
            # (score, word_index/frequency)
            LRtn.append((
                int(idx),
                float(score),
                self.word_index_to_word(int(idx))
            ))
        LRtn.sort(key=lambda i: i[1], reverse=True)
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
            for freq, score, cand in L:
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
        for freq, score, cand in LCands:
            if freq < 500:  # Very high frequency tokens actually yield remarkably low quality results
                continue

            if freq > max(eng_freq, 2000) * 6:
                # It should be pretty unlikely that the word_index in English
                # is an order of magnitude higher in the other language

                # (OPEN ISSUE: Perhaps it could be argued, very common words
                #  in one language may not be the best candidate for
                #  uncommon words?)
                LUnlikely.append((freq, score, cand))
            else:
                LLikely.append((freq, score, cand))

        print("==Likely candidates==")
        do_print(LLikely)
        print("==Unlikely candidates==")
        do_print(LUnlikely)


if __name__ == '__main__':
    from pos_tagger.fasttext_support.aligned.align_sentences import \
        align_sentences
    ENGLISH = AlignedVectors(f'{BASE_PATH}/wiki.en.align.vec')

    print(
        align_sentences(ENGLISH, ENGLISH,
                        'I eat cake for breakfast nope',
                        'Cake I eat breakfast for')
    )

    CHINESE = AlignedVectors(f'{BASE_PATH}/wiki.zh.align.vec')
    #FRENCH = AlignedVectors(f'{BASE_PATH}/wiki.fr.align.vec')

    def print_me(en_text, cn_text, other_inst):
        from_tokens, to_tokens = align_sentences(
            ENGLISH, other_inst, en_text, cn_text
        )
        for item in from_tokens:
            print(item)
        print()
        for item in to_tokens:
            print(item)
        print()

    print_me(
        'It is used for large meetings and conventions.',
        '用作大型和正式会议的举办。', CHINESE
    )
    print_me(
        'May I eat that cake ?',
        '我可以吃那个蛋糕吗？', CHINESE
    )
    print_me(
        'For example, watching TV or going for a swim.',
        '興趣是看電影跟慢跑，擅長游泳。', CHINESE
    )
    print_me(
        'Leo started swimming in 1996.',
        '1996年 王一梅開始練習游泳。', CHINESE
    )
    print_me(
        'Peacekeeping was not the solution, but a means to an end.',
        '维和并不是解决办法，而是达到目的的手段。', CHINESE
    )
    #print_me(
    #    'It is a powerful neighbour.',
    #    'Il est là et c\'est un voisin puissant.', FRENCH
    #)
    #print_me(
    #    "There's no cure for death.",
    #    "Il n'y a pas de remède à la mort.", FRENCH
    #)

    #INDONESIAN = AlignedVectors(f'{BASE_PATH}/wiki.id.align.vec')
    #GERMAN = AlignedVectors(f'{BASE_PATH}/wiki.de.align.vec')

    while True:
        find_me = input("Enter a word to find:")
        for x in range(100):
            try:
                ENGLISH.print_translations(CHINESE, find_me)
                #ENGLISH.print_translations(INDONESIAN, find_me)
                #ENGLISH.print_translations(GERMAN, find_me)
            except KeyError:
                print(f"{find_me} was not found")
