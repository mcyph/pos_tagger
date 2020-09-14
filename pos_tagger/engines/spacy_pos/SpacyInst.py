from pos_tagger.consts import CubeItem
from pos_tagger.engines.EngineInstance import EngineInstance


# Can also use 'sm', 'lg' or 'md' for
# higher accuracy with vector embeddings
SPACY_SIZE = 'sm'

DSpacy = {
    'en': f'en_core_web_{SPACY_SIZE}',
    'de': f'de_core_news_{SPACY_SIZE}',
    'fr': f'fr_core_news_{SPACY_SIZE}',
    'es': f'es_core_news_{SPACY_SIZE}',
    'pt': f'pt_core_news_{SPACY_SIZE}',
    'it': f'it_core_news_{SPACY_SIZE}',
    'nl': f'nl_core_news_{SPACY_SIZE}',
    'el': f'el_core_news_{SPACY_SIZE}',
    'nb': f'nb_core_news_{SPACY_SIZE}',
    'lt': f'lt_core_news_{SPACY_SIZE}',

    'zh': f'zh_core_web_{SPACY_SIZE}',
    'zh_Hant': f'zh_core_web_{SPACY_SIZE}',
    'da': f'da_core_news_{SPACY_SIZE}',
    'ja': f'ja_core_news_{SPACY_SIZE}',
    'pl': f'pt_core_news_{SPACY_SIZE}',
    'ro': f'ro_core_news_{SPACY_SIZE}',
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


if __name__ == '__main__':
    from os import system

    system(f'python3 -m spacy download zh_core_web_{SPACY_SIZE}')
    system(f'python3 -m spacy download da_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download nl_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download en_core_web_{SPACY_SIZE}')
    system(f'python3 -m spacy download fr_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download de_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download el_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download it_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download ja_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download lt_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download nb_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download pl_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download pt_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download ro_core_news_{SPACY_SIZE}')
    system(f'python3 -m spacy download es_core_news_{SPACY_SIZE}')
