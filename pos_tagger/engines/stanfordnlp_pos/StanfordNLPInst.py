from pos_tagger.engines.EngineInstance import EngineInstance
from pos_tagger.consts import CubeItem


class StanfordNLPInst(EngineInstance):
    def __init__(self, iso, use_gpu=False):
        import stanfordnlp
        self.nlp = stanfordnlp.Pipeline(
            lang=iso,
            processors="tokenize,pos,depparse,lemma,mwt",
            use_gpu=False  # self.use_gpu
        )
        EngineInstance.__init__(self, iso, use_gpu)

    def get_L_sentences(self, s):
        LRtn = []
        doc = self.nlp(s)
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
    for x in range(100):
        for LSentence in (
            #get_L_sentences('en', 'The quick brown fox jumps over the lazy dog.')
            get_L_sentences('zh', ' CAMP試驗能將Streptococcus agalactiae區分出來。 ')
        ):
            for token in LSentence:
                print(token)
