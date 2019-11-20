# Add support for https://spacy.io/
from _thread import allocate_lock
from pos_tagger.consts import CubeItem


USE_SPACY_UDPIPE = True
DUDPipeLangs = {}


if USE_SPACY_UDPIPE:

    from pos_tagger.engines.spacy_udpipe_langs import get_D_udpipe_langs
    DUDPipeLangs = get_D_udpipe_langs()


def get_L_supported_isos():
    L = [
        'de',
        'el',
        'en',
        'es',
        'fr',
        'it',
        'nl',
        'pt',
        #'xx',
    ]
    if USE_SPACY_UDPIPE:
        L += list(DUDPipeLangs.keys())
    return L


DNLP = {}
check_nlp_lock = allocate_lock()


def get_L_sentences(iso, s):
    with check_nlp_lock:
        if not iso in DNLP:
            if iso in DUDPipeLangs:
                # UDPipe ones seem to be more accurate a lot of the time
                import spacy_udpipe
                DNLP[iso] = spacy_udpipe.load(iso)
            else:
                import spacy
                DNLP[iso] = spacy.load(iso)

        nlp = DNLP[iso]

    LRtn = []
    LTokens = list(nlp(s))

    for i, token in enumerate(LTokens, start=1):
        #print(token.head, token.whitespace_, token.n_lefts, token.n_rights, token.sentiment, token.shape, dir(token), dir(token.head))
        # TODO: FIX space after!!!
        LRtn.append(CubeItem(
            index=i,
            word=token.text,
            lemma=token.lemma_,
            upos=token.pos_,
            xpos=token.tag_, # TODO: WHERE SHOULD THIS GO??
            attrs=token.dep,
            head=int(token.head.i)+1 if token.head else '',
            label='',
            space_after='_' if token.whitespace_ else 'SpaceAfter=No'
        ))

    return [LRtn]


if __name__ == '__main__':
    for x in range(100):
        doc = ' Here, men are promoted and women can visit the catalog.'

        for LSentence in get_L_sentences('en', doc):
            for token in LSentence:
                print(token)

                # token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                #                         token.shape_, token.is_alpha, token.is_stop
