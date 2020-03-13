import termcolor
from _thread import allocate_lock
from pos_tagger.consts import CubeItem, DPOSColours
from pos_tagger.engines.EngineBase import EngineBase
from pos_tagger.licenses.get_ud_license import get_L_ud_iso_codes, get_L_ud_licenses

INCLUDE_NON_COMMERCIAL = False
INCLUDE_GPL = False


# From https://github.com/adobe/NLP-Cube/blob/master/scripts/package_ud_models.py
model_tuples = [ #folder, language_code, embedding code
    ("UD_Afrikaans-AfriBooms","af","af"),
    ("UD_Ancient_Greek-PROIEL","grc","grc"),
    ("UD_Arabic-PADT","ar","ar"),
    ("UD_Armenian-ArmTDP","hy","hy"),
    ("UD_Basque-BDT","eu","eu"),
    ("UD_Bulgarian-BTB","bg","bg"),
    ("UD_Buryat-BDT","bxr","bxr"),
    ("UD_Catalan-AnCora","ca","ca"),
    ("UD_Chinese-GSD","zh","zh"),
    ("UD_Croatian-SET","hr","hr"),
    ("UD_Czech-PDT","cs","cs"),
    ("UD_Danish-DDT","da","da"),
    ("UD_Dutch-Alpino","nl","nl"),
    ("UD_English-EWT","en","en"),
    ("UD_Estonian-EDT","et","et"),
    ("UD_Finnish-TDT","fi","fi"),
    ("UD_French-GSD","fr","fr"),
    ("UD_Galician-CTG","gl","gl"),
    ("UD_German-GSD","de","de"),
    ("UD_Gothic-PROIEL","got","got"),
    ("UD_Greek-GDT","el","el"),
    ("UD_Hebrew-HTB","he","he"),
    ("UD_Hindi-HDTB","hi","hi"),
    ("UD_Hungarian-Szeged","hu","hu"),
    ("UD_Indonesian-GSD","id","id"),
    ("UD_Irish-IDT","ga","ga"),
    ("UD_Italian-ISDT","it","it"),
    ("UD_Japanese-GSD","ja","ja"),
    ("UD_Kazakh-KTB","kk","kk"),
    ("UD_Korean-GSD","ko","ko"),
    ("UD_Kurmanji-MG","kmr","ku"),
    ("UD_Latin-ITTB","la","la"),
    ("UD_Latvian-LVTB","lv","lv"),
    ("UD_North_Sami-Giella","sme","se"),
    ("UD_Norwegian-Bokmaal","no_bokmaal","no"),
    ("UD_Norwegian-Nynorsk","no_nynorsk","nn"),
    ("UD_Old_Church_Slavonic-PROIEL","cu","cu"),
    ("UD_Persian-Seraji","fa","fa"),
    ("UD_Polish-LFG","pl","pl"),
    ("UD_Portuguese-Bosque","pt","pt"),
    ("UD_Romanian-RRT","ro","ro"),
    ("UD_Russian-SynTagRus","ru","ru"),
    ("UD_Serbian-SET","sr","sr"),
    ("UD_Slovak-SNK","sk","sk"),
    ("UD_Slovenian-SSJ","sl","sl"),
    ("UD_Spanish-AnCora","es","es"),
    ("UD_Swedish-LinES","sv","sv"),
    ("UD_Swedish-Talbanken","sv","sv"),
    ("UD_Turkish-IMST","tr","tr"),
    ("UD_Ukrainian-IU","uk","uk"),
    ("UD_Upper_Sorbian-UFAL","hsb","hsb"),
    ("UD_Urdu-UDTB","ur","ur"),
    ("UD_Uyghur-UDT","ug","ug"),
    ("UD_Vietnamese-VTB","vi","vi")
]

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


def _filter_licenses():
    # Remove non-commercial/gpl models
    for ud_license in get_L_ud_licenses(
        include_non_commercial=True,
        include_gpl=True
    ):
        for model_name, _, iso in model_tuples:
            model_name = model_name[3:]
            if ud_license.model_name == model_name:
                if 'NC' in ud_license.license and not INCLUDE_NON_COMMERCIAL:
                    #print("NON COMMERCIAL:", ud_license)
                    del DSupportedISOs[iso]
                if ud_license.license.startswith('GPL') and not INCLUDE_GPL:
                    #print("GPL:", ud_license)
                    del DSupportedISOs[iso]
_filter_licenses()


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
