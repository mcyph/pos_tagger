import termcolor
from _thread import allocate_lock
from pos_tagger.consts import CubeItem, DPOSColours
from pos_tagger.engines.EngineBase import EngineBase


DSupportedISOs = dict((
    ('af', 'af'),
    ('ar', 'ar'),
    ('bg', 'bg'),
    ('bxr', 'bxr'),
    ('ca', 'ca'),
    ('cs', 'cs'),
    ('cu', 'cu'),
    ('da', 'da'),
    ('de', 'de'),
    ('el', 'el'),
    ('en', 'en'),
    ('es', 'es'),
    ('et', 'et'),
    ('eu', 'eu'),
    ('fa', 'fa'),
    ('fi', 'fi'),
    ('fr', 'fr'),
    ('ga', 'ga'),
    ('gl', 'gl'),
    ('got', 'got'),
    ('grc', 'grc'),
    ('he', 'he'),
    ('hi', 'hi'),
    ('hr', 'hr'),
    ('hsb', 'hsb'),
    ('hu', 'hu'),
    ('hy', 'hy'),
    ('id', 'id'),
    ('it', 'it'),
    ('ja', 'ja'),
    ('kk', 'kk'),
    ('kmr', 'kmr'),
    ('ko', 'ko'),
    ('la', 'la'),
    ('lv', 'lv'),
    ('nl', 'nl'),
    ('nob', 'no_bokmaal'),
    ('nno', 'no_nynorsk'),
    ('pt', 'pt'),
    ('ro', 'ro'),
    ('ru', 'ru'),
    #('sk', 'sk'),
    ('sl', 'sl'),
    ('sme', 'sme'),
    ('sr', 'sr'),
    ('sv', 'sv'),
    ('tr', 'tr'),
    ('ug', 'ug'),
    ('uk', 'uk'),
    ('ur', 'ur'),
    ('vi', 'vi'),
    ('zh', 'zh'),
    ('zh_Hant', 'zh')  # CHECK ME!
))


DCubeInsts = {}
check_nlp_lock = allocate_lock()


class CubeNLPPOS(EngineBase):
    TYPE = 0
    NEEDS_GPU = True

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso in DSupportedISOs

    def get_L_supported_isos(self):
        return list(sorted(DSupportedISOs.keys()))

    def get_L_sentences(self, iso, s):
        with check_nlp_lock:
            try:
                cube_inst = self.get_from_cache(iso)
            except KeyError:
                assert iso in DSupportedISOs
                # Note the GPU argument - it may be worth putting
                # this on the PC with a more powerful one
                from cube.api import Cube
                cube_inst = Cube(
                    verbose=True,
                    use_gpu=self.use_gpu
                )
                # Chinese doesn't seem to work well with version 1.0, [[??? <-  I think this means 1.1??]]
                # despite the scores not showing much difference
                cube_inst.load(
                    DSupportedISOs[iso],
                    version='1.0' if iso == 'zh' else '1.1'
                )

                self.add_to_cache(iso, cube_inst)

        LSentences = cube_inst(s)

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
