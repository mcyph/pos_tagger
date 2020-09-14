from _thread import allocate_lock
from os.path import exists
from functools import lru_cache
from collections import OrderedDict

from pos_tagger.engines.EngineProcess import EngineProcess
from pos_tagger.abstract_base_classes.POSTaggersBase import POSTaggersBase
from pos_tagger.engines.cubenlp_pos.CubeNLPPOS import CubeNLPPOS
from pos_tagger.engines.jieba_pos.JiebaPOS import JiebaPOS
from pos_tagger.engines.pykomoran_pos.PyKomoranPOS import PyKomoranPOS
from pos_tagger.engines.pythainlp_pos.PyThaiNLPPOS import PyThaiNLPPOS
from pos_tagger.engines.pyvi_pos.PyViPOS import PyViPOS
from pos_tagger.engines.spacy_pos.SpacyPOS import SpacyPOS
from pos_tagger.engines.stanfordnlp_pos.StanfordNLPPOS import StanfordNLPPOS

from pos_tagger.fasttext_support.aligned.AlignedVectors import \
    BASE_PATH, AlignedVectors
from pos_tagger.fasttext_support.aligned.align_sentences import \
    align_sentences
from pos_tagger.consts import AlignedCubeItem


_lock = allocate_lock()
_av_lock = allocate_lock()

JOIN_CHARS = '\n฀'  # 2 chars here - second unassigned in unicode


class POSTaggers(POSTaggersBase):
    def __init__(self, use_gpu=False,
                 num_engines_in_cache=14):

        self.use_gpu = use_gpu
        self.num_engines_in_cache = num_engines_in_cache

        self.DGetLSentences = self.__get_D_get_L_sentences()
        self.SSupportedISOs = set(self.get_L_supported_isos())
        self.DPOSEngineCache = _LimitedSizeDict(
            size_limit=num_engines_in_cache
        )
        self.DAVCache = _LimitedSizeDict(
            size_limit=num_engines_in_cache
        )
        self._DSentenceCache = {}

    #============================================================#
    #                     POS Tagger-Related                     #
    #============================================================#

    def get_from_cache(self, typ, iso):
        assert _lock.locked()
        return self.DPOSEngineCache[typ, iso]

    def add_to_cache(self, typ, iso, inst):
        assert _lock.locked()
        self.DPOSEngineCache[typ, iso] = inst

    def __get_from_av_cache(self, iso):
        with _av_lock:
            try:
                return self.DAVCache[iso]
            except KeyError:
                av = AlignedVectors(f'{BASE_PATH}/wiki.{iso}.align.vec')
                self.DAVCache[iso] = av
                return av

    def __get_D_get_L_sentences(self):
        pykomoran_pos = PyKomoranPOS(self)
        jieba_pos = JiebaPOS(self)
        pythainlp_pos = PyThaiNLPPOS(self)
        pyvi_pos = PyViPOS(self)
        spacy_pos = SpacyPOS(self)
        cubenlp_pos = CubeNLPPOS(self)
        stanfordnlp_pos = StanfordNLPPOS(self)

        # {iso: fn, ...}
        DGetLSentences = {
            'ko': pykomoran_pos,
            'th': pythainlp_pos,
            'vi': pyvi_pos,
            'zh': jieba_pos,
            'zh_Hant': jieba_pos
        }

        # Add spaCy first, as it's quite
        # fast and has a lot of features
        # (even if not always the most accurate)
        for _iso in spacy_pos.get_L_supported_isos():
            if _iso in DGetLSentences:
                continue
            DGetLSentences[_iso] = spacy_pos

        if self.use_gpu and False:
            #if _iso == 'zh':
            #    # We'll try traditional chinese as well
            #    assert not 'zh_Hant' in DGetLSentences
            #    DGetLSentences['zh_Hant'] = _closure(
            #        stanfordnlp_pos.get_L_sentences, _iso
            #    )

            # CubeNLP supports lots of languages, but is slow+needs GPU
            # acceleration that I can't have (with CUDA), so put it last
            for _iso in cubenlp_pos.get_L_supported_isos():
                if _iso in DGetLSentences:
                    continue
                DGetLSentences[_iso] = cubenlp_pos

            # Next add stanford NLP
            for _iso in stanfordnlp_pos.get_L_supported_isos():
                if _iso in DGetLSentences:
                    continue
                DGetLSentences[_iso] = stanfordnlp_pos

        return DGetLSentences

    def get_L_supported_isos(self):
        return list(sorted(self.DGetLSentences.keys()))

    def is_iso_supported(self, iso):
        return iso in self.SSupportedISOs

    def get_L_sentences(self, iso, s):
        # Send through a proxy process (using multiprocessing)
        # as many of these POS engines interfere with each
        # other otherwise (especially when using the GPU!)
        DCache = self._DSentenceCache.setdefault(iso, {})
        if len(DCache) > 500:
            self._DSentenceCache[iso] = {}

        cached_item = DCache.get(s)
        if cached_item is not None:
            return cached_item

        r = self._get_L_sentences(iso, s)
        DCache[s] = r
        return r

    def __enqueued_loop(self):
        from pos_tagger.consts import CubeItem

        while True:
            while self.wake_up.q.size():
                self.wake_up_q.get()

            for iso, i_set in self._DQueues.items():
                joined = JOIN_CHARS.join(i_set)

                r = self._get_L_sentences(iso, joined)

                r_out = []
                current = []
                index_offset = 0

                for i in r:
                    if i.word.strip() == JOIN_CHARS[-1]:
                        current = []
                        index_offset = i.index-1

                    elif JOIN_CHARS[-1] in i.word:
                        current.append(
                            CubeItem(
                                index=i.index - index_offset,
                                word=i.word.split(JOIN_CHARS[-1])[0],
                                lemma=i.lemma,
                                upos=i.upos,
                                xpos=i.xpos,
                                attrs=i.attrs,
                                head=i.head,
                                label=i.label,
                                space_after=i.space_after
                            )
                        )
                        r_out.append(current)
                        index_offset = i.index-1
                        current = [
                            CubeItem(
                                index=i.index - index_offset,
                                word=i.word.split(JOIN_CHARS[-1])[-1],
                                lemma=i.lemma,
                                upos=i.upos,
                                xpos=i.xpos,
                                attrs=i.attrs,
                                head=i.head,
                                label=i.label,
                                space_after=i.space_after
                            )
                        ]
                    else:
                        current.append(i)

                if current:
                    r_out.append(current)

                i_set.clear()

    def enqueue_sentence(self, iso, sentence):
        cached = self._DSentenceCache.get(iso, {}).get(sentence)
        if cached is not None:
            # Already done!
            return

        if not iso in self._DQueues:
            self._DQueues[iso] = set()
        self._DQueues[iso].add(sentence)
        self.wake_up_q.put(None)

    def _get_L_sentences(self, iso, s):
        with _lock:
            try:
                engine_process = self.get_from_cache(
                    self.DGetLSentences[iso].TYPE, iso
                )
            except KeyError:
                inst_class = self.DGetLSentences[iso].INST_CLASS
                engine_process = EngineProcess(
                    inst_class, iso,
                    use_gpu=self.use_gpu
                )
                self.add_to_cache(
                    self.DGetLSentences[iso].TYPE,
                    iso, engine_process
                )
            r = engine_process.get_L_sentences(s)
            return r

    #============================================================#
    #                      fastText-Related                      #
    #============================================================#

    def is_alignment_supported(self, from_iso, to_iso):
        return (
            self.is_iso_supported(from_iso) and
            self.is_iso_supported(to_iso) and
            exists(f'{BASE_PATH}/wiki.{from_iso}.align.vec') and
            exists(f'{BASE_PATH}/wiki.{to_iso}.align.vec')
        )

    def get_aligned_sentences(self,
                              from_iso, to_iso,
                              from_s, to_s):

        # TODO: Return AlignedCubeItem's
        LFromCubeItems = self.get_L_sentences(from_iso, from_s)[0]
        LToCubeItems = self.get_L_sentences(to_iso, to_s)[0]

        from_av = self.__get_from_av_cache(from_iso)
        to_av = self.__get_from_av_cache(to_iso)

        LFromAligned, LToAligned = align_sentences(
            from_av, to_av,
            LFromCubeItems,
            LToCubeItems
        )

        LFromRtn = []
        for from_cube_item, from_aligned in zip(LFromCubeItems, LFromAligned):
            D = from_cube_item._asdict()
            D.update(from_aligned._asdict())
            LFromRtn.append(AlignedCubeItem(**D))

        LToRtn = []
        for to_cube_item, to_aligned in zip(LToCubeItems, LToAligned):
            D = to_cube_item._asdict()
            D.update(to_aligned._asdict())
            LToRtn.append(AlignedCubeItem(**D))
        return LFromRtn, LToRtn

    def get_similar_words(self, iso, word, n=30):
        av = self.__get_from_av_cache(iso)
        vec = av.get_vector_for_word(word)
        return av.get_similar_words(vec, n)

    def get_translations(self, from_iso, to_iso, s):
        from_av = self.__get_from_av_cache(from_iso)
        to_av = self.__get_from_av_cache(to_iso)
        return from_av.get_translations(to_av, s)

    def fasttext_get_num_words(self, iso):
        return len(self.__get_from_av_cache(iso))

    def get_fasttext_words(self, iso, exclude_high_freq=True):
        # Discard the vector
        return list([(int(i[0]), i[1]) for i in
                     self.__get_from_av_cache(iso).iter(exclude_high_freq)])


class _LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        # https://stackoverflow.com/questions/2437617/how-to-limit-the-size-of-a-dictionary
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                k, v = self.popitem(last=False)
                if hasattr(v, 'destroy'):
                    v.destroy()


if __name__ == '__main__':
    pt = POSTaggers(use_gpu=True)
    from_, to_ = pt.get_aligned_sentences(
        'en', 'en',
        'The quick brown fox jumps',
        'The brown quick jumps fox'
    )

    for item in from_:
        print(item)
    print()
    for item in to_:
        print(item)
    print()
