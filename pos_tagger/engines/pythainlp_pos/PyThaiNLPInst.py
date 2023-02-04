from pos_tagger.engines.EngineInstance import EngineInstance
from pos_tagger.consts import CubeItem, DOrchardToUDPOS


class PyThaiNLPInst(EngineInstance):
    def __init__(self, iso, use_gpu=False):
        assert iso == 'th'
        EngineInstance.__init__(self, iso, use_gpu)

    def get_L_sentences(self, s):
        from pythainlp import word_tokenize
        from pythainlp.tag import pos_tag  # , pos_tag_sents

        tokens_list = word_tokenize(s)
        LPosTags = pos_tag(tokens_list)

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
