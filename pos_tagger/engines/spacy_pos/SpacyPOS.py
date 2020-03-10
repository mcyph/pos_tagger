from os import system
from pos_tagger.engines.spacy_pos.SpacyPOSBase import SpacyPOSBase


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


class SpacyPOS(SpacyPOSBase):
    TYPE = 5

    def get_L_supported_isos(self):
        L = list(DSpacy.keys())
        return L

    def _download_engine(self, iso):
        engine_name = DSpacy[iso]
        system(f"python3 -m spacy download {engine_name}")

    def _get_model(self, iso):
        import spacy
        return spacy.load(DSpacy[iso])
