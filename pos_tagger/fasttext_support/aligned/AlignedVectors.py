"""
TODO: Use https://fasttext.cc/docs/en/aligned-vectors.html

* Find definitions
* Find words in parallel sentences
* Perhaps could also allow training my own models?
* Output relative word probabilities
"""
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


if __name__ == '__main__':
    from pos_tagger.fasttext_support.aligned.align_sentences import \
        align_sentences
    BASE_PATH = '/mnt/docs/dev/data/fast_text/aligned_word_vectors'
    ENGLISH = AlignedVectors(f'{BASE_PATH}/wiki.en.align.vec')

    print(
        align_sentences(ENGLISH, ENGLISH,
                        'I eat cake for breakfast nope',
                        'Cake I eat breakfast for')
    )

    CHINESE = AlignedVectors(f'{BASE_PATH}/wiki.zh.align.vec')

    def print_me(en_text, cn_text):
        from_tokens, to_tokens = align_sentences(
            ENGLISH, CHINESE, en_text, cn_text
        )
        for item in from_tokens:
            print(item)
        print()
        for item in to_tokens:
            print(item)

    print_me(
        'It is used for large meetings and conventions.',
        '用作大型和正式会议的举办。'
    )
    print()
    print_me(
        'May I eat that cake ?',
        '我可以吃那个蛋糕吗？'
    )

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
