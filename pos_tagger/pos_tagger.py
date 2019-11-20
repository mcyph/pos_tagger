from pos_tagger.engines import jieba_pos, spacy_pos, pykomoran_pos, pyvi_pos, stanfordnlp_pos, cubenlp_pos, \
    pythainlp_pos
from pos_tagger.client_server.POSTaggerClient import POSTaggerClient


USE_POS_TAGGER_SERVER = True


def __get_D_get_L_sentences():
    # {iso: fn, ...}
    DGetLSentences = {
        'ko': pykomoran_pos.get_L_sentences,
        'th': pythainlp_pos.get_L_sentences,
        'vi': pyvi_pos.get_L_sentences,
        'zh': lambda s: jieba_pos.get_L_sentences('zh', s),
        'zh_Hant': lambda s: jieba_pos.get_L_sentences('zh_Hant', s)
    }

    def _closure(original_fn, iso):
        def fn(s):
            return original_fn(iso, s)
        return fn

    # Add spacy first, as it's quite
    # fast and has a lot of features
    for _iso in spacy_pos.get_L_supported_isos():
        if _iso in DGetLSentences:
            continue
        DGetLSentences[_iso] = _closure(
            spacy_pos.get_L_sentences, _iso
        )

    if False:
        # Next add stanford NLP
        for _iso in stanfordnlp_pos.get_L_supported_isos():
            if _iso in DGetLSentences:
                continue
            DGetLSentences[_iso] = _closure(
                stanfordnlp_pos.get_L_sentences, _iso
            )

            if _iso == 'zh':
                # We'll try traditional chinese as well
                assert not 'zh_Hant' in DGetLSentences
                DGetLSentences['zh_Hant'] = _closure(
                    stanfordnlp_pos.get_L_sentences, _iso
                )

        # CubeNLP supports lots of languages, but is slow+needs GPU
        # acceleration that I can't have (with CUDA), so put it last
        for _iso in cubenlp_pos.get_L_supported_isos():
            if _iso in DGetLSentences:
                continue
            DGetLSentences[_iso] = _closure(
                cubenlp_pos.get_L_sentences, _iso
            )

    return DGetLSentences

__DGetLSentences = __get_D_get_L_sentences()


def get_L_supported_isos():
    return list(sorted(__DGetLSentences.keys()))
__SSupportedISOs = set(get_L_supported_isos())


def is_iso_supported(iso):
    return iso in __SSupportedISOs

__pos_tagger_client = None


def get_L_sentences(iso, s, use_pos_tagger_server=True):
    if use_pos_tagger_server and USE_POS_TAGGER_SERVER:
        global __pos_tagger_client
        if not __pos_tagger_client:
            __pos_tagger_client = POSTaggerClient()
        r = __pos_tagger_client.get_L_sentences(iso, s)
        #print(r)
        return r
    else:
        return __DGetLSentences[iso](s)
