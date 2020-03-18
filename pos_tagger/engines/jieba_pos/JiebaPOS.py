# TODO: Support the jieba segmentation module!
from pos_tagger.engines.EngineBase import EngineBase
from pos_tagger.engines.jieba_pos.JiebaInst import JiebaInst


class JiebaPOS(EngineBase):
    TYPE = 1
    NEEDS_GPU = False
    INST_CLASS = JiebaInst

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso in ('zh', 'zh_Hant')

    def get_L_supported_isos(self):
        return ['zh', 'zh_Hant']

