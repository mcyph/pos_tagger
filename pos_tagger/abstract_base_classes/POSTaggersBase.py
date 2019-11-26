from abc import ABC, abstractmethod


class POSTaggersBase(ABC):
    @abstractmethod
    def get_L_supported_isos(self):
        """

        :return:
        """
        pass

    @abstractmethod
    def is_iso_supported(self, iso):
        """

        :param iso:
        :return:
        """
        pass

    @abstractmethod
    def get_L_sentences(self, iso, s, use_pos_tagger_server=True):
        """

        :param iso:
        :param s:
        :param use_pos_tagger_server:
        :return:
        """
        pass
