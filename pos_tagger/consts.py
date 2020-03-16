from collections import namedtuple


CubeItem = namedtuple('CubeItem', [
    'index',
    'word',
    'lemma',
    'upos',
    'xpos',
    'attrs',
    'head',
    'label',
    'space_after'
])

AlignedCubeItem = namedtuple('AlignedCubeItem', [
    'index',
    'word',
    'lemma',
    'upos',
    'xpos',
    'attrs',
    'head',
    'label',
    'space_after',

    # Alignment info
    'source_index',
    'target_index',
    'source_text',
    'target_text',
    'score'
])

DPOSColours = {
    'AUX': 'yellow',  # e.g. can
    'ADP': 'yellow',  # adposition, e.g. of
    'PART': 'yellow',  # e.g. tidak
    'PREP': 'yellow',
    'MODPART': 'yellow',
    'NEG': 'yellow',
    'NSUF': 'yellow',
    'VSUF': 'yellow',
    'ADJSUF': 'yellow',
    'ROOT': 'yellow',

    'NOUN': 'red',  # noun
    'PROPN': 'red',  # proper noun
    'PRON': 'red',  # pronoun
    'DET': 'red',  # e.g. the - usually goes before nouns/pronouns, so will use the same colour
    'NOUNC': 'red',
    'UNOUN': 'red',
    'NOUNL': 'red',

    'MORPH': 'red', # ???
    'IDIOM': 'red',
    'ABBR': 'red',
    'PNOUN': 'red',
    'ORGNOUN': 'red',
    'VNOUN': 'red',

    'ADV': 'cyan',

    'VERB': 'blue',
    'SECVERB': 'blue',

    # May pay to group these together, as e.g. first/pertama
    # can described as ADJ/NUM respectively
    'NUM': 'magenta',
    'ADJ': 'magenta',
    'POSADJ': 'magenta',
    'ADJNOHENBU': 'magenta',
    'QUANT': 'magenta',
    'TIME': 'magenta',

    'PUNCT': 'white',
    'SYM': 'white',
    'X': 'white',  # other,
    'SPACE': 'white',

    # FIXME!
    'PARAL': 'white',
    'ONOM': 'white',

    'SCONJ': 'green',  # subordinating conjunction
    'CONJ': 'green',  # coordinating conjunction
    'CCONJ': 'green',  # maybe CONJ isn't used for coordinating conjunction in cubenlp?
}

DPOS = {
    # From cubenlp
    'ADJ': 'adjective',
    'ADP': 'adposition',
    'ADV': 'adverb',
    'AUX': 'auxiliary verb',
    'CONJ': 'conjunction',  # NOTE: The original UD used CONJ for coordinating conjunction, but I'm using it for conjunctions of any kind, as jieba doesn't always say which is which
    'CCONJ': 'coordinating conjunction',
    'DET': 'determiner',
    'INTJ': 'interjection',
    'NOUN': 'noun',
    'NUM': 'numeral',
    'PART': 'particle',
    'PRON': 'pronoun',
    'PROPN': 'proper noun',
    'PUNCT': 'punctuation',
    'SCONJ': 'subordinating conjunction',
    'SYM': 'symbol',
    'VERB': 'verb',
    'X': 'other',

    # Added from pyvi
    'NOUNC': 'noun classifier',
    'UNOUN': 'unit noun',

    # Added from Jieba
    'PARAL': 'paralinguistic feature',
    'NOUNL': 'noun of locality',
    'MORPH': 'morpheme',
    'IDIOM': 'idiom',
    'ABBR': 'abbreviation',
    'PNOUN': 'place noun',
    'ORGNOUN': 'organisation/group noun',
    'ONOM': 'onomatopoeia',
    'PREP': 'preposition',
    'QUANT': 'quantity',
    'TIME': 'time word',
    'VNOUN': 'verb used as noun',
    'MODPART': 'modal particle',
    'ADJNOHENBU': 'adjective which can\'t have 很 or 不 before it',
    #'ENG': 'english',

    # Added from pykomoran
    'SECVERB': 'secondary verb',
    'POSADJ': 'positive adjective',
    'NEG': 'negator',
    'NSUF': 'noun derived suffix',
    'VSUF': 'verb derived suffix',
    'ADJSUF': 'adjective derived suffix',
    'ROOT': 'root'
}

# Though orchard for pythainlp is more specific,
# I think it's probably too specific for my purposes.
DOrchardToUDPOS = {}

for line in """
NOUN	NOUN
NCMN	NOUN
NTTL	NOUN
CNIT	NOUN
CLTV	NOUN
CMTR	NOUN
CFQC	NOUN
CVBL	NOUN
VACT	VERB
VSTA	VERB
PROPN	PROPN
NPRP	PROPN
ADJ	ADJ
NONM	ADJ
VATT	ADJ
DONM	ADJ
ADV	ADV
ADVN	ADV
ADVI	ADV
ADVP	ADV
ADVS	ADV
INT	INTJ
PRON	PRON
PPRS	PRON
PDMN	PRON
PNTR	PRON
DET	DET
DDAN	DET
DDAC	DET
DDBQ	DET
DDAQ	DET
DIAC	DET
DIBQ	DET
DIAQ	DET
NUM	NUM
NCNM	NUM
NLBL	NUM
DCNM	NUM
AUX	AUX
XVBM	AUX
XVAM	AUX
XVMM	AUX
XVBB	AUX
XVAE	AUX
ADP	ADP
RPRE	ADP
CCONJ	CCONJ
JCRG	CCONJ
SCONJ	SCONJ
PREL	SCONJ
JSBR	SCONJ
JCMP	SCONJ
PART	PART
FIXN	PART
FIXV	PART
EAFF	PART
EITT	PART
AITT	PART
NEG	PART
PUNCT	PUNCT
PUNC	PUNCT
""".strip().split('\n'):

    orchard, ud = line.split('\t')
    DOrchardToUDPOS[orchard] = ud
