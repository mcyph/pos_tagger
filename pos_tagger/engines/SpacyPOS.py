# Add support for https://spacy.io/
from _thread import allocate_lock
from pos_tagger.consts import CubeItem
from pos_tagger.engines.EngineBase import EngineBase

USE_SPACY_UDPIPE = False

if USE_SPACY_UDPIPE:
    from pos_tagger.engines.spacy_udpipe_langs import get_D_udpipe_langs
    DUDPipeLangs = get_D_udpipe_langs()
else:
    DUDPipeLangs = {}

DNLP = {}
check_nlp_lock = allocate_lock()


class SpacyPOS(EngineBase):
    TYPE = 5
    NEEDS_GPU = False

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def download_engine(self):
        TODO

    def is_iso_supported(self, iso):
        return iso in self.get_L_supported_isos()

    def get_L_supported_isos(self):
        L = [
            'en',
            'de',
            'fr',
            'es',
            'pt',
            'it',
            'nl',
            'el',
            'nb',
            'lt',
        ]
        if USE_SPACY_UDPIPE:
            L += list(DUDPipeLangs.keys())
        return L

    def get_L_sentences(self, iso, s):
        with check_nlp_lock:
            try:
                nlp = self.get_from_cache(iso)
            except KeyError:
                import spacy
                if self.use_gpu:
                    # TODO: Support different GPU ids??
                    spacy.require_gpu()

                if iso in DUDPipeLangs:
                    # UDPipe ones seem to be more accurate a lot of the time
                    import spacy_udpipe
                    nlp = spacy_udpipe.load(iso)
                else:
                    nlp = spacy.load(iso)
                self.add_to_cache(iso, nlp)

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
    from spacy import download
    """
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    python -m spacy download fr_core_news_sm
    python -m spacy download es_core_news_sm
    python -m spacy download pt_core_news_sm
    python -m spacy download it_core_news_sm
    python -m spacy download nl_core_news_sm
    python -m spacy download el_core_news_sm
    python -m spacy download nb_core_news_sm
    python -m spacy download lt_core_news_sm
    
    Check https://spacy.io/usage/models
    """

    for x in range(100):
        doc = ' Here, men are promoted and women can visit the catalog.'

        for LSentence in get_L_sentences('en', doc):
            for token in LSentence:
                print(token)

                # token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                #                         token.shape_, token.is_alpha, token.is_stop
