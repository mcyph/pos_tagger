from pos_tagger.POSTaggers import POSTaggers
from network_tools.posix_shm_sockets.SHMServer import SHMServer, json_method


class POSTaggerServer(SHMServer):
    def __init__(self):
        """
        A server which e.g. allows putting the POS Tagger on
        a server which has a GPU for POS tagging acceleration
        """
        self.pos_taggers = POSTaggers()
        SHMServer.__init__(self, DCmds={
            'get_L_sentences': self.get_L_sentences,
            'is_iso_supported': self.is_iso_supported,
            'get_L_supported_isos': self.get_L_supported_isos,
        }, port=40519)

    @json_method
    def get_L_sentences(self, iso, s):
        return self.pos_taggers.get_L_sentences(
            iso, s
        )

    @json_method
    def is_iso_supported(self, iso):
        return self.pos_taggers.is_iso_supported(iso)

    @json_method
    def get_L_supported_isos(self):
        return self.pos_taggers.get_L_supported_isos()
