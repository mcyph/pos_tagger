from pos_tagger.POSTaggers import POSTaggers

from speedysvc.rpc_decorators import json_method
from speedysvc.client_server.base_classes.ServerMethodsBase import ServerMethodsBase


class CPUPOSTaggerServer(ServerMethodsBase):
    port = 40519
    name = 'postag'

    def __init__(self, logger_client, use_gpu=False):
        """
        A server which e.g. allows putting the POS Tagger on
        a server which has a GPU for POS tagging acceleration
        """
        ServerMethodsBase.__init__(self, logger_client)
        self.pos_taggers = POSTaggers(use_gpu=use_gpu)

    @json_method
    def get_L_sentences(self, iso, s):
        return self.pos_taggers.get_L_sentences(iso, s)

    @json_method
    def is_iso_supported(self, iso):
        return self.pos_taggers.is_iso_supported(iso)

    @json_method
    def get_L_supported_isos(self):
        return self.pos_taggers.get_L_supported_isos()

    @json_method
    def is_alignment_supported(self, from_iso, to_iso):
        return self.pos_taggers.is_alignment_supported(
            from_iso, to_iso
        )

    @json_method
    def get_aligned_sentences(self,
                              from_iso, to_iso,
                              from_s, to_s):
        return self.pos_taggers.get_aligned_sentences(
            from_iso, to_iso, from_s, to_s
        )

    @json_method
    def get_similar_words(self, iso, word, n=30):
        return self.pos_taggers.get_similar_words(iso, word, n)

    @json_method
    def get_translations(self, from_iso, to_iso, s):
        return self.pos_taggers.get_translations(from_iso, to_iso, s)

    @json_method
    def fasttext_get_num_words(self, iso):
        return self.pos_taggers.fasttext_get_num_words(iso)

    @json_method
    def get_fasttext_words(self, iso, exclude_high_freq=True):
        return self.pos_taggers.get_fasttext_words(iso, exclude_high_freq)


class GPUPOSTaggerServer(CPUPOSTaggerServer):
    def __init__(self, logger_client, use_gpu=True):
        CPUPOSTaggerServer.__init__(self, logger_client, use_gpu)
