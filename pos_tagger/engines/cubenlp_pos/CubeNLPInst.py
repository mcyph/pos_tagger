import termcolor
from pos_tagger.consts import CubeItem, DPOSColours
from pos_tagger.engines.EngineInstance import EngineInstance


class CubeNLPInst(EngineInstance):
    def __init__(self, iso, use_gpu=False):
        from pos_tagger.engines.cubenlp_pos.CubeNLPPOS import \
            DSupportedISOs
        assert self.iso in DSupportedISOs

        # Note the GPU argument - it may be worth putting
        # this on the PC with a more powerful one
        from cube.api import Cube
        cube_inst = self.cube_inst = Cube(
            verbose=True,
            use_gpu=self.use_gpu
        )
        # Chinese doesn't seem to work well with version 1.0,
        # [[??? <-  I think this means 1.1??]]
        # despite the scores not showing much difference
        cube_inst.load(
            DSupportedISOs[self.iso],
            version='1.0' if iso == 'zh' else '1.1'
        )
        EngineInstance.__init__(self, iso, use_gpu)

    def get_L_sentences(self, s):
        LSentences = self.cube_inst(s)

        LRtn = []
        for LSentence in LSentences:
            LItem = []
            for entry in LSentence:
                LItem.append(CubeItem(
                    index=int(entry.index),
                    word=entry.word,
                    lemma=entry.lemma,
                    upos=entry.upos,
                    xpos=entry.xpos,
                    attrs=entry.attrs,
                    head=int(entry.head) if str(entry.head).isnumeric() else '', # CHECK ME!
                    label=str(entry.label),
                    space_after=entry.space_after
                ))
            LRtn.append(LItem)
        return LRtn

    def print_pos(self, iso, s):
        LSentences = self.get_L_sentences(iso, s)

        for sentence in LSentences:
            for entry in sentence:
                termcolor.cprint(entry.word, DPOSColours[entry.upos], end=' ')
            print()

        for LSentence in LSentences:
            for entry in LSentence:
                print(entry)
            print("")


if __name__ == '__main__':
    if True:
        found_nl = False
        for iso in CubeNLPPOS.get_L_supported_isos(None):
            if iso not in ('nno', 'nnb'):
                continue

            from cube.api import Cube  # import the Cube object

            cube = Cube(verbose=True)  # initialize it
            cube.load(DSupportedISOs[iso])

    print_pos('id', 'Tahap pertama konflik ini dapat disebut "Perang Kemerdekaan Belanda".')
    print_pos('en', 'The first phase of the conflict can be considered the Dutch War of Independence.')

    print_pos('id', 'Saya tidak dapat memakan ini.')
    print_pos('en', 'I can\'t eat this.')

    print_pos('zh', '猴子高兴，实验人员也高兴。')
    print_pos('en', 'The monkeys were happy and the experimenters were happy.')

    #for iso in get_L_supported_isos():
    #    if iso < 'he':
    #        continue

    #    print("CREATING FOR ISO:", iso)
    #    print_pos(iso, '.')

    for x in range(100):
        for LSentence in (
            get_L_sentences('en', 'The quick brown fox jumps over the lazy dog.')
        ):
            for token in LSentence:
                print(token)
