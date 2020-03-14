import importlib

_DStopWordsCache = {}

_LPossibleISOs = [i.strip() for i in """
af 
ar 
bg 
bn 
ca 
cs 
da 
de 
el 
en 
es 
et 
eu 
fa 
fi 
fr 
ga 
he 
hi 
hr 
hu 
id 
is 
it 
ja 
kn 
ko 
lb 
lt 
lv 
mr 
nb 
nl 
pl 
pt 
ro 
ru 
si 
sk 
sl 
sq 
sr 
sv 
ta 
te 
th 
tl 
tr 
tt 
uk 
ur 
vi 
xx 
yo 
zh 
""".strip().split('\n')]


def get_S_stop_words_for_iso(iso):
    # Will use spaCy's stopwords - don't think there's
    # much point doing duplicated work
    # Hardcoding ISOs to prevent potential vulnerabilities
    try:
        assert iso in _LPossibleISOs
        mod = importlib.import_module('spacy.lang.%s' % iso)
        _DStopWordsCache[iso] = mod.STOP_WORDS
    except (ImportError, AttributeError, AssertionError):
        _DStopWordsCache[iso] = set()
    return _DStopWordsCache[iso]
