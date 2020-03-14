from pos_tagger.fasttext_support.tokenizers.ICUTokenizer import WordTokenizer

import importlib
import numpy as np
from sys import maxsize
from _thread import allocate_lock
from collections import namedtuple
from pos_tagger.fasttext_support.aligned.punct_ws_symbols import \
    REMOVE_PUNCT_WS_SYMBOLS_TABLE
from pos_tagger.fasttext_support.aligned.stop_words import \
    get_S_stop_words_for_iso


#====================================================================#
#                        Sentence Alignment                          #
#====================================================================#


AlignedItem = namedtuple('AlignedItem', [
    'source_index', 'target_index',
    'source_text', 'target_text',
    'score'
])


def align_sentences(from_inst, to_inst,
                    from_s, to_s,
                    ignore_stop_words=True,
                    tolerance=1.18): # 1.24

    if ignore_stop_words:
        SFromStopWords = get_S_stop_words_for_iso(from_inst.iso)
        SToStopWords = get_S_stop_words_for_iso(to_inst.iso)
    else:
        SFromStopWords = set()
        SToStopWords = set()

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
        if not from_token.translate(REMOVE_PUNCT_WS_SYMBOLS_TABLE).strip() or \
                from_token.lower() in SFromStopWords:
            # Ignore stop words/punctuation/whitespace/symbols
            from_vec = None
        else:
            try:
                from_vec = from_inst.get_vector_for_word(
                    from_token.lower()
                )
            except KeyError:
                from_vec = None
        LFromVecs.append(from_vec)

    for y, to_token in enumerate(LToTokens):
        if not to_token.translate(REMOVE_PUNCT_WS_SYMBOLS_TABLE).strip() or \
                to_token.lower() in SToStopWords:
            to_vec = None
        else:
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
    smallest = maxsize

    while len(DFromToMap) != min(len(LToVecs), len(LFromVecs)):
        idx1, idx2 = smallest_indices(
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

    # Output aligned tokens (AlignedItem's)
    # Won't output whitespace for now
    LFromRtn = []
    for x, from_token in enumerate(LFromTokens):
        if x in DFromToMap and DFromToMap[x][-1] <= smallest*tolerance:
            other_idx, score = DFromToMap[x]
            LFromRtn.append(AlignedItem(x+1, other_idx+1,
                                        from_token, LToTokens[other_idx],
                                        score))
        else:
            LFromRtn.append(AlignedItem(x+1, None,
                                        from_token, None,
                                        None))

    LToRtn = []
    for y, to_token in enumerate(LToTokens):
        if y in DToFromMap and DToFromMap[y][-1] <= smallest*tolerance:
            other_idx, score = DToFromMap[y]
            LToRtn.append(AlignedItem(y+1, other_idx+1,
                                      to_token, LFromTokens[other_idx],
                                      score))
        else:
            LToRtn.append(AlignedItem(y+1, None,
                                      to_token, None,
                                      None))
    return LFromRtn, LToRtn


#====================================================================#
#                          Miscellaneous                             #
#====================================================================#


_tokenizer_lock = allocate_lock()
_DWordTokenizers = {}


def get_tokens(iso, s):
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
                iso.replace('-', '_'
            ).replace(
                '|', '_'
            ))
        return _DWordTokenizers[iso].get_segments(s)


def smallest_indices(ary, n):
    """
    Returns the n smallest indices from a numpy array.
    """
    flat = ary.flatten()
    indices = np.argpartition(flat, n)[:n] # CHECK ME!!
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, ary.shape)
