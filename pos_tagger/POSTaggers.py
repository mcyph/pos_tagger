from pos_tagger.abstract_base_classes.POSTaggersBase import POSTaggersBase


class POSTaggers(POSTaggersBase):
    def __init__(self):
        self.DGetLSentences = self.__get_D_get_L_sentences()
        self.SSupportedISOs = set(self.get_L_supported_isos())

    def __get_D_get_L_sentences(self):
        from pos_tagger.engines import jieba_pos, spacy_pos, \
            pykomoran_pos, pyvi_pos, stanfordnlp_pos, \
            cubenlp_pos, pythainlp_pos

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

    def get_L_supported_isos(self):
        return list(sorted(self.DGetLSentences.keys()))

    def is_iso_supported(self, iso):
        return iso in self.SSupportedISOs

