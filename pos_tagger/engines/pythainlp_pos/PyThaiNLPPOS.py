from pos_tagger.engines.EngineBase import EngineBase
from pos_tagger.engines.pythainlp_pos.PyThaiNLPInst import \
    PyThaiNLPInst


class PyThaiNLPPOS(EngineBase):
    TYPE = 3
    NEEDS_GPU = False
    INST_CLASS = PyThaiNLPInst

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso == 'th'

    def get_L_supported_isos(self):
        return ['th']
