"""
TODO: Use https://fasttext.cc/docs/en/aligned-vectors.html

* Find definitions
* Find words in parallel sentences
* Perhaps could also allow training my own models?
* Output relative word probabilities
"""
import numpy as np


def get_D_word_embeddings(path):
    DWords = {}
    DFreqs = {}

    with open(path, 'r', encoding='utf-8') as f:
        for x, line in enumerate(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            DWords[word] = coefs
            DFreqs[word] = x  # More common words might be higher up(?)
            if x % 1000 == 0:
                print(x)
    return DFreqs, DWords


DEnglishFreqs, DEnglish = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.en.align.vec'
)
DChineseFreqs, DChinese = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.zh.align.vec'
)
DIndonesianFreqs, DIndonesian = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.id.align.vec'
)
DGermanFreqs, DGerman = get_D_word_embeddings(
    '/mnt/docs/dev/data/fast_text/aligned_word_vectors/wiki.de.align.vec'
)


def get_L_cands(DVectors, find_me):
    LCands = []
    for key, value in DVectors.items():
        LCands.append((abs(np.sum(np.abs(find_me - value))), key))
    LCands.sort()
    return LCands[:30]


def get_diff_cands(DFromVectors, DToVectors,
                   from_vec, to_vec):

    # Note range seems to be -1.0->+1.0
    not_in_to = (from_vec-to_vec)
    not_in_from = (to_vec-from_vec)

    min_ = min(to_vec.min(), from_vec.min())
    max_ = max(to_vec.max(), from_vec.max())

    not_in_to = np.interp(
        not_in_to,
        (not_in_to.min(), not_in_to.max()),
        (min_, max_)  # WARNING!
    )
    not_in_from = np.interp(
        not_in_from,
        (not_in_from.min(), not_in_from.max()),
        (min_, max_)
    )

    print("NOT IN TO:", not_in_to)
    print("FROM VEC:", from_vec)
    print("TO VEC:", to_vec)

    return (
        get_L_cands(DFromVectors, not_in_from),
        get_L_cands(DFromVectors, not_in_to),
        get_L_cands(DToVectors, not_in_from),
        get_L_cands(DToVectors, not_in_to),
    )


def print_translations(DWordVectors, DWordFreqs, s):
    LCands = get_L_cands(DWordVectors, DEnglish[s])

    L30k = []
    LGreater = []

    def do_print(L):
        for score, cand in L:
            print(
                score, cand,
                f"English Freq: "
                f"{DEnglishFreqs[s]}/{len(DEnglishFreqs)}",
                f"Other Freq: "
                f"{DWordFreqs[cand]}/{len(DWordFreqs)}"
            )

    # Really uncommon words are
    # lower quality and likely junk
    for score, cand in LCands:
        freq = DWordFreqs[cand]
        if freq < 30000:
            L30k.append((score, cand))
        else:
            LGreater.append((score, cand))

    print("==In 30000 Most Common Words==")
    do_print(L30k)
    print("==Uncommon Words==")
    do_print(LGreater)


if __name__ == '__main__':
    #print_translations('cat')
    #print_translations('king')
    #print_translations('queen')

    while True:
        find_me = input("Enter a word to find:")
        try:
            print_translations(DChinese, DChineseFreqs, find_me)
            print_translations(DIndonesian, DIndonesianFreqs, find_me)
            print_translations(DGerman, DGermanFreqs, find_me)
        except KeyError:
            print(f"{find_me} was not found")
