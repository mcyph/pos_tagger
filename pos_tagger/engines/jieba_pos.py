# TODO: Support the jieba segmentation module!
#from jieba_hant import posseg as posseg_hant
from pos_tagger.consts import CubeItem

# TODO: USE https://gist.github.com/luw2007/6016931   !!!


DJiebaToCube = {
    'ag': 'ADJ', # HACK
    'a': 'ADJ',
    'ad': 'ADV',
    'an': 'NOUN', # HACK: adjective used as noun
    'b': 'ATTR', # ADD ME!
    'c': 'CONJ',
    'dg': 'PARAL', # ADD ME: Paralinguistic feature
    'd': 'ADV',
    'df': 'ADV', # ???
    'e': 'INTJ',
    'eng': 'X', # HACK: make english categorized into "other"
    'f': 'NOUNL', # ADD ME: Noun of locality
    'g': 'MORPH', # ADD ME: Morpheme
    'h': 'NOUN', # FIXME!!! ===========================
    'i': 'IDIOM',
    'j': 'ABBR',
    'k': 'NOUN', # FIXME!!! ===========================
    'l': 'IDIOM', # HACK!
    'm': 'NUM',
    'mg': 'X', # ???
    'mq': 'X', # ???
    'ng': 'NOUN', # HACK
    'n': 'NOUN',
    'nr': 'PROPN',
    'ns': 'PNOUN',
    'nt': 'ORGNOUN',
    'nz': 'PROPN', # HACK!
    'nrt': 'NOUN', # ????
    'o': 'ONOM',
    'p': 'PREP',
    'q': 'QUANT',
    'r': 'PRON',
    'rz': 'PRON', # ????
    'rr': 'PROM', # ???
    's': 'PUNCT', # Space
    'tg': 'TIME', # HACK
    't': 'TIME',
    'u': 'AUX',
    'ul': 'AUX', # ?????
    'uj': 'AUX', #  ?????
    'uz': 'AUX', # ???
    'ug': 'AUX', # ???
    'ud': 'AUX', # ???
    'uv': 'AUX', # ????
    'vg': 'VERB', # HACK
    'v': 'VERB',
    'vd': 'ADV', # Adverbial form of verb
    'vn': 'VNOUN',
    'w': 'PUNCT',
    'x': 'PUNCT',
    'y': 'MODPART',
    'yg': 'MODPART', # CHECK ME!!!
    'z': 'ADJNOHENBU',
    'zg': 'ADJ', #  ??????
    'un': 'X',
}

"""
Ag	形语素	形容词性语素。形容词代码为 a，语素代码ｇ前面置以A。
a	形容词	取英语形容词 adjective的第1个字母。
ad	副形词	直接作状语的形容词。形容词代码 a和副词代码d并在一起。
an	名形词	具有名词功能的形容词。形容词代码 a和名词代码n并在一起。
b	区别词	取汉字“别”的声母。
c	连词	取英语连词 conjunction的第1个字母。
dg 	副语素	副词性语素。副词代码为 d，语素代码ｇ前面置以D。
d	副词	取 adverb的第2个字母，因其第1个字母已用于形容词。
e	叹词	取英语叹词 exclamation的第1个字母。
f	方位词	取汉字“方”
g	语素	绝大多数语素都能作为合成词的“词根”，取汉字“根”的声母。
h	前接成分	取英语 head的第1个字母。
i	成语	取英语成语 idiom的第1个字母。
j	简称略语	取汉字“简”的声母。
k	后接成分	 
l	习用语	习用语尚未成为成语，有点“临时性”，取“临”的声母。
m	数词	取英语 numeral的第3个字母，n，u已有他用。
Ng	名语素	名词性语素。名词代码为 n，语素代码ｇ前面置以N。
n	名词	取英语名词 noun的第1个字母。
nr	人名	名词代码 n和“人(ren)”的声母并在一起。
ns	地名	名词代码 n和处所词代码s并在一起。
nt	机构团体	“团”的声母为 t，名词代码n和t并在一起。
nz	其他专名	“专”的声母的第 1个字母为z，名词代码n和z并在一起。
o	拟声词	取英语拟声词 onomatopoeia的第1个字母。
p	介词	取英语介词 prepositional的第1个字母。
q	量词	取英语 quantity的第1个字母。
r	代词	取英语代词 pronoun的第2个字母,因p已用于介词。
s	处所词	取英语 space的第1个字母。
tg	时语素	时间词性语素。时间词代码为 t,在语素的代码g前面置以T。
t	时间词	取英语 time的第1个字母。
u	助词	取英语助词 auxiliary
vg	动语素	动词性语素。动词代码为 v。在语素的代码g前面置以V。
v	动词	取英语动词 verb的第一个字母。
vd	副动词	直接作状语的动词。动词和副词的代码并在一起。
vn	名动词	指具有名词功能的动词。动词和名词的代码并在一起。
w	标点符号	 
x	非语素字	非语素字只是一个符号，字母 x通常用于代表未知数、符号。
y	语气词	取汉字“语”的声母。
z	状态词	取汉字“状”的声母的前一个字母。
un	未知词	不可识别词及用户自定义词组。取英文Unkonwn首两个字母。(非北大标准，CSW分词中定义)
"""


def get_L_sentences(iso, s):
    LRtn = []
    from jieba_fast import posseg as posseg_hans

    if iso == 'zh_Hant':
        callable = posseg_hans.cut
    elif iso == 'zh':
        callable = posseg_hans.cut
    else:
        raise Exception("Unknown ISO code: %s" % iso)

    L = list(callable(s))

    for i, (segment, pos) in enumerate(L, start=1):
        LRtn.append(CubeItem(
            index=i,
            word=segment.replace('_', ' '),
            lemma=segment.replace('_', ' '),
            upos=DJiebaToCube[pos],
            xpos='',
            attrs='',
            head='',
            label='',
            space_after='SpaceAfter=No'
        ))

    return [LRtn]


if __name__ == '__main__':
    for LSentence in get_L_sentences(
        'zh', '非语素字只是一个符号，字母 x通常用于代表未知数、符号。'
    ):
        for cube_item in LSentence:
            print(cube_item)

    for LSentence in get_L_sentences(
        'zh_Hant', '非语素字只是一个符号，字母 x通常用于代表未知数、符号。'
    ):
        for cube_item in LSentence:
            print(cube_item)

