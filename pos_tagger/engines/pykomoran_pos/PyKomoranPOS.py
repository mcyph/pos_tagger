# PyKomoran 불러오기
from _thread import get_ident
from pos_tagger.consts import CubeItem
from pos_tagger.engines.EngineBase import EngineBase



"""
 Common nouns
Proper noun
Dependent noun NNB
Pronoun NP
Investigation NR
Verb VV
Adjectives VA
Assistance VX
Positive designator VCP
Negative designator VCN
Tubular MM
General Adverbs MAG
Connection adjacency MAJ
Interjection IC
Main investigation JKS
Supplementary investigation JKC
Tubular Screening JKG
Objective investigation
Sub-shot investigation JKB
Investigation
Judged by JKQ
Assistant JX
Connection survey JC
Fresh Horse EP
Terminating EF
Connection mother EC
Noun-type malleable ETH
Tubular malleable mother ETM
Message prefix XPN
Noun Derivative Suffix XSN
Derivative Suffix XSV
Adjective Derivative Suffix XSA
Root XR
Period, question mark, exclamation point SF
Comma, gown, colon, hatched SP
Quotation marks, parenthesis, line mark SS
Ellipsis SE
Attachment mark (wave, hidden, missing) SO
Foreign language SL
Chinese characters SH
Other Symbols (Logical Math Symbols, Currency Symbols) SW
Noun Estimation Category NF
Proverb Estimation Category NV
Number SN
Inparable Category NA 
"""


DKomoranToUD = {
    'NNG': 'NOUN', #'normal noun',
    'NNP': 'NOUN', # 'unique noun',
    'NNB': 'NOUN', # 'dependent noun',
    'NP': 'PRON', # 'pronoun',
    'NR': 'NUM', # 'number',
    'VV': 'VERB', # 'verb',
    'VA': 'ADJ', # 'adjective',
    'VX': 'SECVERB', # 'secondary verb', note me
    'VCP': 'POSADJ', # 'positive adjective', note me
    'VCN': 'NEG', # 'negative specifier',
    'MM': 'DET', # 'determiner',
    'MAG': 'ADV', # 'General Adverbs',
    'MAJ': 'ADV', # 'Interconnect Adverbs',
    'IC': 'INTJ', # 'Interjection',
    'JKS': 'PART', # 'Critical Investigation',
    'JKC': 'PART', # 'Survey',
    'JKG': 'PART', # 'Tubular Investigation',
    'JKO': 'PART', # 'Purpose survey',
    'JKB': 'PART', # 'Subshot survey',
    'JKV': 'PART', # 'Rock investigation',
    'JKQ': 'PART', # 'Personal survey',
    'JX': 'PART', # 'Assisted survey',
    'JC': 'PART', # 'Connection survey',
    'EP': 'X', # 'premature ending', ???
    'EF': 'X', # 'ending ending', ???
    'EC': 'X', # 'connecting ending', ???
    'ETN': 'X', # 'noun malleable mother', ???
    'ETM': 'X', # 'tubular malleable mother',
    'XPN': 'PREF', # 'commit prefix',
    'XSN': 'NSUF', # 'noun derived suffix',
    'XSV': 'VSUF', # 'verb derived suffix',
    'XSA': 'ADJSUF', # 'adjective derived suffix',
    'XR': 'ROOT', # 'root',
    'SF': 'SYM', # 'Period, question mark, exclamation point',
    'SP': 'SYM', # 'comma, center point, colon, hatch',
    'SS': 'SYM', # 'quotation, bracket, ellipse',
    'SE': 'SYM', # 'ellipsis',
    'SO': 'SYM', # 'additional mark (Wave, hidden, missing)'
    'SW': 'SYM', # ???
    'SL': 'SYM', # ???
    'SN': 'SYM',
    'SH': 'SYM',
    'NA': 'X', # ??? I think this means "can't categorize" or similar
}


# Komoran 객체 생성
DKomoran = {}
MAX_CACHED_INSTS = 2  # Not sure how many we can have without
                      # java.lang.OutOfMemoryError: Java heap space
                      # occurring


class PyKomoranPOS(EngineBase):
    TYPE = 2
    NEEDS_GPU = False

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso == 'ko'

    def get_L_supported_isos(self):
        return ['ko']

    def get_L_sentences(self, iso, s):
        assert iso == 'ko'

        LRtn = []

        if len(DKomoran) > MAX_CACHED_INSTS:
            global DKomoran
            DKomoran = {}

        if not get_ident() in DKomoran:
            from PyKomoran import Komoran
            DKomoran[get_ident()] = Komoran("EXP")  # Memory warning!! =========================================

        komoran = DKomoran[get_ident()]
        LTokenList = list(komoran.get_token_list(s))

        for i, token in enumerate(LTokenList):
            segment = s[token.begin_index:token.end_index]
            pos = token.pos

            next_token = (
                LTokenList[i+1]
                if i != len(LTokenList)-1
                else None
            )
            if next_token:
                if token.end_index == next_token.begin_index:
                    space_after = 'SpaceAfter=No'
                else:
                    space_after = '_'
            else:
                space_after = 'SpaceAfter=No'

            LRtn.append(CubeItem(
                index=i,
                word=segment.replace('_', ' '),
                lemma=segment.replace('_', ' '),
                upos=DKomoranToUD[pos],
                xpos='',
                attrs='',
                head='',
                label='',
                space_after=space_after # FIXME!
            ))

        return [LRtn]


if __name__ == '__main__':
    # 분석할 문장 준비
    str_to_analyze = "① 대한민국은 민주공화국이다. ② 대한민국의 주권은 국민에게 있고, 모든 권력은 국민으로부터 나온다."
    for LSentence in get_L_sentences(str_to_analyze):
        for item in LSentence:
            print(item)
