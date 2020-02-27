from pos_tagger.POSTaggers import POSTaggers

from speedysvc.rpc_decorators import json_method
from speedysvc.client_server.base_classes.ServerMethodsBase import ServerMethodsBase


class POSTaggerServer(ServerMethodsBase):
    port = 40519
    name = 'postag'

    def __init__(self, logger_client):
        """
        A server which e.g. allows putting the POS Tagger on
        a server which has a GPU for POS tagging acceleration
        """
        ServerMethodsBase.__init__(self, logger_client)
        self.pos_taggers = POSTaggers()

    @json_method
    def get_L_sentences(self, iso, s):
        return self.pos_taggers.get_L_sentences(iso, s)

    @json_method
    def is_iso_supported(self, iso):
        return self.pos_taggers.is_iso_supported(iso)

    @json_method
    def get_L_supported_isos(self):
        return self.pos_taggers.get_L_supported_isos()
