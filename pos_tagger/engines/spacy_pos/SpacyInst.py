from pos_tagger.consts import CubeItem
from pos_tagger.engines.EngineInstance import EngineInstance


# Can also use 'lg' or 'md' for
# higher accuracy with vector embeddings
USE_MODEL_KIND = 'sm'


DSpacy = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
    'es': 'es_core_news_sm',
    'pt': 'pt_core_news_sm',
    'it': 'it_core_news_sm',
    'nl': 'nl_core_news_sm',
    'el': 'el_core_news_sm',
    'nb': 'nb_core_news_sm',
    'lt': 'lt_core_news_sm',
}


class SpacyInst(EngineInstance):
    def __init__(self, iso, use_gpu=False):
        import spacy
        if use_gpu:
            # TODO: Support different GPU ids??
            spacy.require_gpu()

        self.nlp = spacy.load(DSpacy[iso])
        EngineInstance.__init__(self, iso, use_gpu)

    def get_L_sentences(self, s):
        LRtn = []
        LTokens = list(self.nlp(s))

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
