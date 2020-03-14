from collections import namedtuple
from sys import maxsize
import numpy as np

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
                DFromToMap[x] = y, a[x, y]
                DToFromMap[y] = x, a[x, y]
            a[x, y] = maxsize

    # Output aligned tokens (AlignedItem's)
    # Won't output whitespace for now
    LFromRtn = []
    for x, from_token in enumerate(LFromTokens):
        if x in DFromToMap:
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
        if y in DToFromMap:
            other_idx, score = DToFromMap[y]
            LToRtn.append(AlignedItem(y+1, other_idx+1,
                                      to_token, LFromTokens[other_idx],
                                      score))
        else:
            LToRtn.append(AlignedItem(y+1, None,
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
