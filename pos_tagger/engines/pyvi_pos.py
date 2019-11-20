# TODO: Add support for Vietnamese via pyvi
#https://pypi.org/project/pyvi/

from pos_tagger.consts import CubeItem


"""
    A - Adjective
    C - Coordinating conjunction
    E - Preposition
    I - Interjection
    L - Determiner
    M - Numeral
    N - Common noun
    Nc - Noun Classifier
    Ny - Noun abbreviation
    Np - Proper noun
    Nu - Unit noun
    P - Pronoun
    R - Adverb
    S - Subordinating conjunction
    T - Auxiliary, modal words
    V - Verb
    X - Unknown
    F - Filtered out (punctuation)
"""

DPOSToCube = {
    'A': 'ADJ',
    'C': 'CCONJ',
    'E': 'PREP', # ADD ME!
    'I': 'INTJ',
    'L': 'DET',
    'M': 'NUM',
    'N': 'NOUN',
    'Nc': 'NOUNC', # ADD ME!
    'Ny': 'ABBR', # ADD ME!
    'Np': 'PROPN',
    'Nu': 'UNOUN', # ADD ME!
    'P': 'PRON',
    'R': 'ADV',
    'S': 'SCONJ',
    'T': 'AUX',
    'V': 'VERB',
    'X': 'X',
    'F': 'PUNCT',
}


def get_L_sentences(s):
    from pyvi import ViTokenizer, ViPosTagger
    LSeg, LPOS = ViPosTagger.postagging(
        ViTokenizer.tokenize(s)
    )

    LRtn = []

    for i, (segment, pos) in enumerate(zip(LSeg, LPOS), start=1):
        LRtn.append(CubeItem(
            index=i,
            word=segment.replace('_', ' '),
            lemma=segment.replace('_', ' '),
            upos=DPOSToCube[pos],
            xpos='',
            attrs='',
            head='',
            label='',
            space_after='_' if i != len(LSeg)-1 else 'SpaceAfter=No'
        ))

    return [LRtn]


if __name__ == '__main__':
    for LSentence in get_L_sentences(
        "Tôi không có thời gian để tham gia tất cả."
    ):
        for cube_item in LSentence:
            print(cube_item)
