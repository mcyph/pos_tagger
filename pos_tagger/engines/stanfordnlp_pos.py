# TODO: Support stanfordnlp
from _thread import allocate_lock
from pos_tagger.consts import CubeItem


def get_L_supported_isos():
    from stanfordnlp.utils.resources import default_treebanks
    return list(default_treebanks.keys())


DNLP = {}
check_nlp_lock = allocate_lock()


def get_L_sentences(iso, s):
    with check_nlp_lock:
        if not iso in DNLP:
            import stanfordnlp
            DNLP[iso] = stanfordnlp.Pipeline(
                lang=iso,
                processors="tokenize,pos,depparse,lemma,mwt",
                use_gpu=False
            )
        nlp = DNLP[iso]

    LRtn = []
    doc = nlp(s)

    for sent in doc.sentences:
        #print(sent, dir(sent))

        LSentence = []
        for word in sent.words:
            #print(word, dir(word))

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
