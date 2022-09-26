from pos_tagger.POSTaggers import POSTaggers

from speedysvc.service_method import service_method


class CPUPOSTaggerServer:
    #port = 40519
    #name = 'postag'

    def __init__(self, use_gpu=False):
        """
        A server which e.g. allows putting the POS Tagger on
        a server which has a GPU for POS tagging acceleration
        """
        self.pos_taggers = POSTaggers(use_gpu=use_gpu)

    @service_method()
    def get_L_sentences(self, iso, s):
        return self.pos_taggers.get_L_sentences(iso, s)

    @service_method()
    def is_iso_supported(self, iso):
        return self.pos_taggers.is_iso_supported(iso)

    @service_method()
    def get_L_supported_isos(self):
        return self.pos_taggers.get_L_supported_isos()

    @service_method()
    def is_alignment_supported(self, from_iso, to_iso):
        return self.pos_taggers.is_alignment_supported(from_iso, to_iso)

    @service_method()
    def get_aligned_sentences(self,
                              from_iso, to_iso,
                              from_s, to_s):
        return self.pos_taggers.get_aligned_sentences(from_iso, to_iso, from_s, to_s)

    @service_method()
    def get_similar_words(self, iso, word, n=30):
        return self.pos_taggers.get_similar_words(iso, word, n)

    @service_method()
    def get_translations(self, from_iso, to_iso, s):
        return self.pos_taggers.get_translations(from_iso, to_iso, s)

    @service_method()
    def fasttext_get_num_words(self, iso):
        return self.pos_taggers.fasttext_get_num_words(iso)

    @service_method()
    def get_fasttext_words(self, iso, exclude_high_freq=True):
        return self.pos_taggers.get_fasttext_words(iso, exclude_high_freq)


class GPUPOSTaggerServer(CPUPOSTaggerServer):
    def __init__(self, logger_client, use_gpu=True):
        CPUPOSTaggerServer.__init__(self, logger_client, use_gpu)
