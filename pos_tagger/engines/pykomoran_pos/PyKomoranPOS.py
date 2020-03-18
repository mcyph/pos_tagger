from pos_tagger.engines.EngineBase import EngineBase
from pos_tagger.engines.pykomoran_pos.PyKomoranInst import \
    PyKomoranInst


class PyKomoranPOS(EngineBase):
    TYPE = 2
    NEEDS_GPU = False
    INST_CLASS = PyKomoranInst

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)
        self.DQueues = {}

    def is_iso_supported(self, iso):
        return iso == 'ko'

    def get_L_supported_isos(self):
        return ['ko']

