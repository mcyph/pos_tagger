from abc import ABC, abstractmethod


class EngineBase(ABC):
    TYPE = None

    def __init__(self, pos_taggers):
        self.pos_taggers = pos_taggers
        self.use_gpu = pos_taggers.use_gpu
        assert self.TYPE is not None

    @abstractmethod
    def is_iso_supported(self, iso):
        pass

    @abstractmethod
    def get_L_supported_isos(self):
        pass

    def get_from_cache(self, iso):
        return self.pos_taggers.get_from_cache(
            self.TYPE, iso
        )

    def add_to_cache(self, iso, inst):
        self.pos_taggers.add_to_cache(
            self.TYPE, iso, inst
        )
