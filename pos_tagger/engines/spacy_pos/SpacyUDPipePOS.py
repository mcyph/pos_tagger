from pos_tagger.engines.spacy_pos.SpacyPOSBase import SpacyPOSBase
from pos_tagger.engines.spacy_pos.spacy_udpipe_langs import get_D_udpipe_langs

DUDPipeLangs = get_D_udpipe_langs()


class SpacyUDPipePOS(SpacyPOSBase):
    TYPE = 6

    def get_L_supported_isos(self):
        return list(DUDPipeLangs.keys())

    def _download_engine(self, iso):
        import spacy_udpipe
        spacy_udpipe.download(iso)

    def _get_model(self, iso):
        import spacy_udpipe
        nlp = spacy_udpipe.load(iso)
        return nlp
