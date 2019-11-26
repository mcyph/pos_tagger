from toolkit.patterns.Singleton import Singleton
from network_tools.posix_shm_sockets.SHMClient import SHMClient
from pos_tagger.abstract_base_classes.POSTaggersBase import POSTaggersBase
from pos_tagger.consts import CubeItem


class POSTaggerClient(POSTaggersBase, Singleton):
    def __init__(self):
        self.client = SHMClient(port=40519)

    def get_L_sentences(self, iso, s):
        LRtn = self.client.send_json(
            'get_L_sentences', [iso, s]
        )
        n_LRtn = []
        for LSentence in LRtn:
            LSentence = [CubeItem(*i) for i in LSentence]
            n_LRtn.append(LSentence)
        return n_LRtn

    def is_iso_supported(self, iso):
        return self.client.send_json('is_iso_supported', [iso])

    def get_L_supported_isos(self):
        return self.client.send_json('get_L_supported_isos', [])


if __name__ == '__main__':
    client = POSTaggerClient()
    print(client.get_L_sentences('en', 'blah'))
