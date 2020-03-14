# Add support for https://spacy.io/
from abc import ABC, abstractmethod
from _thread import allocate_lock
from pos_tagger.consts import CubeItem
from pos_tagger.engines.EngineBase import EngineBase

DNLP = {}
check_nlp_lock = allocate_lock()


class SpacyPOSBase(EngineBase):
    TYPE = None
    NEEDS_GPU = False

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso in self.get_L_supported_isos()

    @abstractmethod
    def get_L_supported_isos(self):
        pass

    def get_L_sentences(self, iso, s):
        with check_nlp_lock:
            try:
                nlp = self.get_from_cache(iso)
            except KeyError:
                import spacy
                if self.use_gpu:
                    # TODO: Support different GPU ids??
                    spacy.require_gpu()

                nlp = self._get_model(iso)
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

    @abstractmethod
    def _download_engine(self, iso):
        pass

    @abstractmethod
    def _get_model(self, iso):
        pass


if __name__ == '__main__':
    for x in range(100):
        doc = ' Here, men are promoted and women can visit the catalog.'

        for LSentence in get_L_sentences('en', doc):
            for token in LSentence:
                print(token)

                # token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                #                         token.shape_, token.is_alpha, token.is_stop
