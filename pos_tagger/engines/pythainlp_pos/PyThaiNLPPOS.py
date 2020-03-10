from pos_tagger.consts import CubeItem, DOrchardToUDPOS
from pos_tagger.engines.EngineBase import EngineBase


class PyThaiNLPPOS(EngineBase):
    TYPE = 3
    NEEDS_GPU = False

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso == 'th'

    def get_L_supported_isos(self):
        return ['th']

    def get_L_sentences(self, iso, s):
        from pythainlp import word_tokenize
        from pythainlp.tag import pos_tag  # , pos_tag_sents

        LTokens = word_tokenize(s)
        LPosTags = pos_tag(LTokens)

        LRtn = []

        for i, (segment, pos) in enumerate(LPosTags):
            LRtn.append(CubeItem(
                index=i,
                word=segment.replace('_', ' '),
                lemma=segment.replace('_', ' '),
                upos=DOrchardToUDPOS[pos],
                xpos='',
                attrs='',
                head='',
                label='',
                space_after='SpaceAfter=No'
            ))

        return [LRtn]


if __name__ == '__main__':
    for LSentence in get_L_sentences(
        "ภาษาตุรุง เป็นภาษาในกลุ่มภาษาไทที่สูญแล้ว เคยมีการพูดในรัฐอัสสัมของประเทศอินเดีย ชาวตุรุงซึ่งมีประชากรประมาณ 30,000 คน ปัจจุบันได้พูดภาษาอัสสัมหรือภาษาSingphoแทน"
    ):
        for cube_item in LSentence:
            print(cube_item)
