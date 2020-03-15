import numpy as np
from sys import maxsize
from _thread import allocate_lock
from collections import namedtuple

from pos_tagger.fasttext_support.aligned.punct_ws_symbols import \
    REMOVE_PUNCT_WS_SYMBOLS_TABLE
from pos_tagger.fasttext_support.aligned.stop_words import \
    get_S_stop_words_for_iso
from pos_tagger.fasttext_support.tokenizers.ICUTokenizer import \
    WordTokenizer
from pos_tagger.consts import CubeItem, AlignedCubeItem, DPOSColours


_AlignedItem = namedtuple('AlignedItem', [
    'source_index', 'target_index',
    'source_text', 'target_text',
    'score'
])

_tokenizer_lock = allocate_lock()
_DWordTokenizers = {}


class _AlignSentences:
    def __init__(self, from_inst, to_inst,
                       from_sentence, to_sentence,
                       ignore_stop_words=True,
                       tolerance=1.5):

        self.from_inst = from_inst
        self.to_inst = to_inst
        # Can be a CubeItem, a list/tuple of tokens, or a string
        self.from_sentence = from_sentence
        self.to_sentence = to_sentence

        self.ignore_stop_words = ignore_stop_words
        self.tolerance = tolerance

        # Get stop words and tokens
        self.SFromStopWords, self.SToStopWords = self.get_stop_words()
        self.LFromTokens, self.LLemFromTokens = \
            self.get_tokens(self.from_inst, self.from_sentence)
        self.LToTokens, self.LLemToTokens = \
            self.get_tokens(self.to_inst, self.to_sentence)

    #======================================================================#
    #                         Process Input Params                         #
    #======================================================================#

    def get_stop_words(self):
        if self.ignore_stop_words:
            SFromStopWords = get_S_stop_words_for_iso(self.from_inst.iso)
            SToStopWords = get_S_stop_words_for_iso(self.to_inst.iso)
        else:
            SFromStopWords = set()
            SToStopWords = set()

        return SFromStopWords, SToStopWords

    def get_tokens(self, inst, sentence):
        """
        Tokenize the two sentences (if they haven't been already)
        If using a POS system which tokenizes separately, it might
        make sense to just use its output, and not do it again

        Returns (words in the form used in the sentence,
                 lemmatized forms as seen in a dictionary
                 [or None if the tokens are the same/don't
                  have the lemmatized forms])
        """
        #print("GET_TOKENS:", inst, sentence)
        if sentence and isinstance(
            sentence[0], (CubeItem, AlignedCubeItem)
        ):
            # TODO: What about the lemmatized forms??
            LRtn = (
                [i.word for i in sentence],
                [i.lemma or i.word for i in sentence]
            )
            if LRtn[0] == LRtn[1]:
                LRtn = (
                    LRtn[0],
                    None
                )
        elif isinstance(sentence, (list, tuple)):
            LRtn = (
                sentence,
                None
            )
        else:
            LRtn = (
                self._get_tokens(inst.iso, sentence),
                None
            )
        return LRtn

    def _get_tokens(self, iso, s):
        if iso in ('zh', 'zh_Hant'):
            from pos_tagger.engines.jieba_pos.JiebaPOS import JiebaPOS
            class DUMMY: use_gpu = False

            return [
                i.word for i in
                JiebaPOS(DUMMY).get_L_sentences(iso, s)[0]
            ]

        with _tokenizer_lock:
            if not iso in _DWordTokenizers:
                # ICU should use a pretty similar scheme to what I use anyway
                # (e.g. zh_Hant for traditional Chinese) hopefully converting
                # - and | to underscores will work in most or all cases.
                _DWordTokenizers[iso] = WordTokenizer(
                    iso.replace('-', '_').replace('|', '_'))
            return _DWordTokenizers[iso].get_segments(s)

    def get_vectors(self, LTokens, SStopWords, inst):
        LRtn = []
        for x, token in enumerate(LTokens):
            if (
                not token.translate(
                    REMOVE_PUNCT_WS_SYMBOLS_TABLE
                ).strip() or
                token.lower() in SStopWords
            ):
                # Ignore stop words/punctuation/whitespace/symbols
                vec = None
            else:
                try:
                    vec = inst.get_vector_for_word(
                        token.lower()
                    )
                except KeyError:
                    vec = None
            LRtn.append(vec)
        return LRtn

    #======================================================================#
    #                       Do the Alignment Itself                        #
    #======================================================================#

    def align_sentences(self):
        LAlignedFrom, LAlignedTo = self._align_sentences(
            self.LFromTokens, self.LToTokens
        )
        if self.LLemFromTokens or self.LLemToTokens:
            LLemAlignedFrom, LLemAlignedTo = self._align_sentences(
                self.LLemFromTokens or self.LFromTokens,
                self.LLemToTokens or self.LToTokens
            )
            return (
                self._combine_scores(LAlignedFrom, LLemAlignedFrom,
                                     self.from_sentence, self.to_sentence),
                self._combine_scores(LAlignedTo, LLemAlignedTo,
                                     self.to_sentence, self.from_sentence)
            )
        else:
            return LAlignedFrom, LAlignedTo

    def _combine_scores(self, LOut1, LOut2, sentence, other_sentence):
        LRtn = []

        for ai1, ai2 in zip(LOut1, LOut2):
            score = ai2.score or ai1.score  # CHECK ME!!!

            # TODO: SHOULD THESE SCORES BE COMBINED DURING THE ALIGNMENT STAGE??

            if score is not None:
                if (
                    ai2 is not None and
                    ai1.target_index is not None and
                    ai1.target_index == ai2.target_index and
                    ai1.source_text != ai2.source_text
                ):
                    # If lemmatized form gives same result,
                    # assume more likely correct
                    score *= 0.6

                for use_ai in (ai2, ai1):
                    if (
                        use_ai.target_index is not None and
                        sentence and other_sentence and
                        isinstance(other_sentence[0], CubeItem) and
                        isinstance(sentence[0], CubeItem)
                    ):
                        if (
                            other_sentence[use_ai.target_index].upos ==
                            sentence[use_ai.source_index].upos
                        ):
                            # If POS same, more likely
                            score *= 0.6
                        else:
                            color1 = DPOSColours.get(
                                other_sentence[use_ai.target_index].upos)
                            color2 = DPOSColours.get(
                                sentence[use_ai.source_index].upos)

                            if color1 == color2:
                                # Penalize only slightly
                                # if not exactly the same
                                score *= 1.1
                            else:
                                # If the POS is very different (hence displayed as
                                # a different color) the probability of the match
                                # being correct becomes much lower - disqualify!
                                print("DISQUALIFYING:", other_sentence[use_ai.target_index-1], sentence[use_ai.source_index-1])
                                score *= 1.3

            if score is not None and score > 16 and False: # ADJUST THIS NUMBER as needed!!! Often I've seen 18/19 as being a definitive cutoff
                LRtn.append(_AlignedItem(
                    ai1.source_index, None,
                    ai1.source_text, None,
                    maxsize
                ))
            else:
                LRtn.append(_AlignedItem(
                    ai1.source_index, ai1.target_index,
                    ai1.source_text, ai1.target_text,
                    score
                ))
        return LRtn

    def _align_sentences(self, LFromTokens, LToTokens):
        # Get the vectors for all words we can
        # TODO: MAKE SURE IF E.G. "IT'S" has been separated into it and 's that stop words are checked in the lemmatized forms, too!!! =============================================
        LFromVecs = self.get_vectors(
            LFromTokens, self.SFromStopWords, self.from_inst
        )
        LToVecs = self.get_vectors(
            LToTokens, self.SToStopWords, self.to_inst
        )

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
        smallest = maxsize

        while len(DFromToMap) != min(len(LToVecs), len(LFromVecs)):
            idx1, idx2 = self.smallest_indices(
                a, min(len(LFromTokens), len(LToTokens))
            )

            all_maxsize = True
            for x, y in sorted(zip(idx1, idx2), key=lambda xy: a[xy]):
                if smallest == maxsize:
                    smallest = a[x, y]
                else:
                    assert smallest <= a[x, y]

                if all_maxsize and a[x, y] != maxsize:
                    all_maxsize = False
                elif a[x, y] == maxsize:
                    continue

                if not x in DFromToMap and not y in DToFromMap:
                    DFromToMap[x] = y, a[x, y]
                    DToFromMap[y] = x, a[x, y]
                a[x, y] = maxsize

            if all_maxsize:
                # Prevent an infinite loop
                break

        LFromRtn = self.get_aligned_tokens(
            DFromToMap, smallest,
            LFromTokens, LToTokens
        )
        LToRtn = self.get_aligned_tokens(
            DToFromMap, smallest,
            LToTokens, LFromTokens
        )
        return LFromRtn, LToRtn

    #====================================================================#
    #                      Alignment Utility Fns                         #
    #====================================================================#

    def get_aligned_tokens(self, DMap, smallest, LTokens, LOtherTokens):
        """
        Output aligned tokens (AlignedItem's)
        Won't output whitespace for now
        """
        LRtn = []
        for x, token in enumerate(LTokens):
            if x in DMap and DMap[x][-1] <= smallest * self.tolerance:
                other_idx, score = DMap[x]
                LRtn.append(_AlignedItem(int(x), int(other_idx),
                                         token, LOtherTokens[other_idx],
                                         float(score)))
            else:
                LRtn.append(_AlignedItem(int(x), None,
                                         token, None,
                                         None))
        return LRtn

    def smallest_indices(self, ary, n):
        """
        Returns the n smallest indices from a numpy array.
        """
        max_num = ary.shape[0]*ary.shape[1]
        if max_num == 1:
            return [[0], [0]]

        #n = min(max_num, n)
        flat = ary.flatten()
        indices = np.argpartition(flat, n)[:n] # CHECK ME!!
        indices = indices[np.argsort(flat[indices])]
        return np.unravel_index(indices, ary.shape)


def align_sentences(from_inst, to_inst,
                    from_sentence, to_sentence,
                    ignore_stop_words=True,
                    tolerance=100000.0):

    as_ = _AlignSentences(
        from_inst, to_inst,
        from_sentence, to_sentence,
        ignore_stop_words, tolerance
    )
    return as_.align_sentences()
