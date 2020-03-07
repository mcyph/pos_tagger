from _thread import allocate_lock
from pos_tagger.consts import CubeItem
from pos_tagger.engines.EngineBase import EngineBase


DNLP = {}
check_nlp_lock = allocate_lock()


class StanfordNLPPOS(EngineBase):
    TYPE = 6
    NEEDS_GPU = True

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def is_iso_supported(self, iso):
        return iso in self.get_L_supported_isos()

    def get_L_supported_isos(self):
        from stanfordnlp.utils.resources import default_treebanks
        return list(default_treebanks.keys())

    def get_L_sentences(self, iso, s):
        with check_nlp_lock:
            try:
                nlp = self.get_from_cache(iso)
            except KeyError:
                import stanfordnlp
                nlp = stanfordnlp.Pipeline(
                    lang=iso,
                    processors="tokenize,pos,depparse,lemma,mwt",
                    use_gpu=self.use_gpu
                )
                self.add_to_cache(iso, nlp)

        LRtn = []
        doc = nlp(s)
        for sent in doc.sentences:
            LSentence = []
            for word in sent.words:
                LSentence.append(CubeItem(
                    index=int(word.index),
                    word=word.text,
                    lemma=word.lemma,
                    upos=word.upos,
                    xpos=word.xpos,
                    attrs=word.feats,
                    head=int(word.governor) if str(word.governor).isnumeric() else '',  # CHECK ME!
                    label=word.dependency_relation, # THIS IS PROBABKLY INCORRECT!
                    space_after='_'  # FIXME!!!!
                ))
            LRtn.append(LSentence)
        return LRtn


if __name__ == '__main__':
    if False:
        for iso in get_L_supported_isos():
            print("DOWNLOADING:", iso)
            stanfordnlp.download(iso, force=True)
    else:
        for x in range(100):
            for LSentence in (
                #get_L_sentences('en', 'The quick brown fox jumps over the lazy dog.')
                get_L_sentences('zh', ' CAMP試驗能將Streptococcus agalactiae區分出來。 ')
            ):
                for token in LSentence:
                    print(token)
