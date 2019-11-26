from pos_tagger.PosTaggers import PosTaggers
from network_tools.posix_shm_sockets.SHMServer import SHMServer, json_method


class POSTaggerServer(SHMServer):
    def __init__(self):
        """
        A server which e.g. allows putting the POS Tagger on
        a server which has a GPU for POS tagging acceleration
        """
        self.pos_taggers = PosTaggers()
        SHMServer.__init__(self, DCmds={
            'get_L_sentences': self.get_L_sentences
        }, port=40519)

    @json_method
    def get_L_sentences(self, iso, s):
        return self.pos_taggers.get_L_sentences(
            iso, s, use_pos_tagger_server=False
        )
