from abc import ABC, abstractmethod


class EngineInstance:
    def __init__(self, iso, use_gpu=False):
        self.iso = iso
        self.use_gpu = use_gpu

    @abstractmethod
    def get_L_sentences(self, s):
        pass
