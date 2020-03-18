#https://pypi.org/project/pyvi/
from pos_tagger.engines.EngineBase import EngineBase
from pos_tagger.engines.pyvi_pos.PyViInst import PyViInst


class PyViPOS(EngineBase):
    TYPE = 4
    NEEDS_GPU = False
    INST_CLASS = PyViInst

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso == 'vi'

    def get_L_supported_isos(self):
        return ['vi']

