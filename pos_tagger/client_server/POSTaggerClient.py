from pos_tagger.toolkit.patterns.Singleton import Singleton
from speedysvc.client_server.shared_memory.SHMClient import SHMClient
from speedysvc.client_server.base_classes.ClientMethodsBase import ClientMethodsBase

from pos_tagger.abstract_base_classes.POSTaggersBase import POSTaggersBase
from pos_tagger.client_server.POSTaggerServer import CPUPOSTaggerServer as srv
from pos_tagger.consts import CubeItem, AlignedCubeItem


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
        return self.__deserialize_cube_item(LRtn)

    def is_iso_supported(self, iso):
        return self.send(srv.is_iso_supported, [iso])

    def get_L_supported_isos(self):
        return self.send(srv.get_L_supported_isos, [])

    def is_alignment_supported(self, from_iso, to_iso):
        return self.send(srv.is_alignment_supported, [
            from_iso, to_iso
        ])

    def get_aligned_sentences(self,
                              from_iso, to_iso,
                              from_s, to_s):
        L1, L2 = self.send(srv.get_aligned_sentences, [
            from_iso, to_iso, from_s, to_s
        ])
        L1 = self.__deserialize_cube_item([L1])[0]
        L2 = self.__deserialize_cube_item([L2])[0]
        return L1, L2

    def __deserialize_cube_item(self, LRtn):
        n_LRtn = []
        for LSentence in LRtn:
            LSentence = [
                CubeItem(*i) if len(i) == len(CubeItem._fields)
                else AlignedCubeItem(*i)
                for i in LSentence
            ]
            n_LRtn.append(LSentence)
        return n_LRtn

    def get_similar_words(self, iso, word, n=30):
        return self.send(srv.get_similar_words, [
            iso, word, n
        ])

    def get_translations(self, from_iso, to_iso, s):
        return self.send(srv.get_translations, [
            from_iso, to_iso, s
        ])

    def fasttext_get_num_words(self, iso):
        return self.send(srv.fasttext_get_num_words, [iso])


if __name__ == '__main__':
    client = POSTaggerClient()
    print(client.get_L_sentences('en', 'blah'))
    print(client.get_aligned_sentences('en', 'en', 'blah', 'blah'))
