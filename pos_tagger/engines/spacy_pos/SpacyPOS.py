# Add support for https://spacy.io/
from os import system
from pos_tagger.engines.EngineBase import EngineBase
from pos_tagger.engines.spacy_pos.SpacyInst import DSpacy, SpacyInst


class SpacyPOS(EngineBase):
    TYPE = 5
    NEEDS_GPU = False
    INST_CLASS = SpacyInst

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso in self.get_L_supported_isos()

    def get_L_supported_isos(self):
        L = list(DSpacy.keys())
        return L

    def _download_engine(self, iso):
        engine_name = DSpacy[iso]
        system(f"python3 -m spacy download {engine_name}")


if __name__ == '__main__':
    for x in range(100):
        doc = ' Here, men are promoted and women can visit the catalog.'

        for LSentence in get_L_sentences('en', doc):
            for token in LSentence:
                print(token)

                # token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                #                         token.shape_, token.is_alpha, token.is_stop
