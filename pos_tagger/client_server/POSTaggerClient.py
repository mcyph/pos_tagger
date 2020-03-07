from pos_tagger.toolkit.patterns.Singleton import Singleton
from speedysvc.client_server.shared_memory.SHMClient import SHMClient
from speedysvc.client_server.base_classes.ClientMethodsBase import ClientMethodsBase

from pos_tagger.abstract_base_classes.POSTaggersBase import POSTaggersBase
from pos_tagger.client_server.POSTaggerServer import POSTaggerServer as srv
from pos_tagger.consts import CubeItem


class POSTaggerClient(POSTaggersBase,
                      Singleton,
                      ClientMethodsBase
                      ):

    def __init__(self, client_provider=None):
        if client_provider is None:
            client_provider = SHMClient(srv)
        ClientMethodsBase.__init__(self, client_provider)

    def get_L_sentences(self, iso, s):
        LRtn = self.send(
            srv.get_L_sentences, [iso, s]
        )
        n_LRtn = []
        for LSentence in LRtn:
            LSentence = [CubeItem(*i) for i in LSentence]
            n_LRtn.append(LSentence)
        return n_LRtn

    def is_iso_supported(self, iso):
        return self.send(srv.is_iso_supported, [iso])

    def get_L_supported_isos(self):
        return self.send(srv.get_L_supported_isos, [])


if __name__ == '__main__':
    client = POSTaggerClient()
    print(client.get_L_sentences('en', 'blah'))
