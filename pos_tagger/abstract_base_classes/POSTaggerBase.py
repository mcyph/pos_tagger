from abc import ABC, abstractmethod


class POSTaggerBase(ABC):
    @abstractmethod
    def get_L_sentences(self, iso, s):
        """

        :param iso:
        :param s:
        :return:
        """
        pass

