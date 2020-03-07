from collections import OrderedDict

from pos_tagger.abstract_base_classes.POSTaggersBase import POSTaggersBase
from pos_tagger.engines.CubeNLPPOS import CubeNLPPOS
from pos_tagger.engines.JiebaPOS import JiebaPOS
from pos_tagger.engines.PyKomoranPOS import PyKomoranPOS
from pos_tagger.engines.PyThaiNLPPOS import PyThaiNLPPOS
from pos_tagger.engines.PyViPOS import PyViPOS
from pos_tagger.engines.SpacyPOS import SpacyPOS
from pos_tagger.engines.StanfordNLPPOS import StanfordNLPPOS


class POSTaggers(POSTaggersBase):
    def __init__(self, use_gpu=False,
                 num_engines_in_cache=3):

        self.use_gpu = use_gpu
        self.num_engines_in_cache = num_engines_in_cache

        self.DGetLSentences = self.__get_D_get_L_sentences()
        self.SSupportedISOs = set(self.get_L_supported_isos())
        self.DPOSEngineCache = _LimitedSizeDict(
            size_limit=num_engines_in_cache
        )

    def get_from_cache(self, typ, iso):
        return self.DPOSEngineCache[typ, iso]

    def add_to_cache(self, typ, iso, inst):
        self.DPOSEngineCache[typ, iso] = inst

    def __get_D_get_L_sentences(self):
        pykomoran_pos = PyKomoranPOS(self)
        cubenlp_pos = CubeNLPPOS(self)
        jieba_pos = JiebaPOS(self)
        pythainlp_pos = PyThaiNLPPOS(self)
        pyvi_pos = PyViPOS(self)
        spacy_pos = SpacyPOS(self)
        stanfordnlp_pos = StanfordNLPPOS(self)

        # {iso: fn, ...}
        DGetLSentences = {
            'ko': pykomoran_pos,
            'th': pythainlp_pos,
            'vi': pyvi_pos,
            'zh': jieba_pos,
            'zh_Hant': jieba_pos
        }

        # Add spacy first, as it's quite
        # fast and has a lot of features
        for _iso in spacy_pos.get_L_supported_isos():
            if _iso in DGetLSentences:
                continue
            DGetLSentences[_iso] = spacy_pos

        if self.use_gpu:
            # Next add stanford NLP
            for _iso in stanfordnlp_pos.get_L_supported_isos():
                if _iso in DGetLSentences:
                    continue
                DGetLSentences[_iso] = stanfordnlp_pos

                #if _iso == 'zh':
                #    # We'll try traditional chinese as well
                #    assert not 'zh_Hant' in DGetLSentences
                #    DGetLSentences['zh_Hant'] = _closure(
                #        stanfordnlp_pos.get_L_sentences, _iso
                #    )

            # CubeNLP supports lots of languages, but is slow+needs GPU
            # acceleration that I can't have (with CUDA), so put it last
            for _iso in cubenlp_pos.get_L_supported_isos():
                if _iso in DGetLSentences:
                    continue
                DGetLSentences[_iso] = cubenlp_pos

        return DGetLSentences

    def get_L_supported_isos(self):
        return list(sorted(self.DGetLSentences.keys()))

    def is_iso_supported(self, iso):
        return iso in self.SSupportedISOs

    def get_L_sentences(self, iso, s):
        return self.DGetLSentences[iso].get_L_sentences(iso, s)


class _LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        # https://stackoverflow.com/questions/2437617/how-to-limit-the-size-of-a-dictionary
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)

