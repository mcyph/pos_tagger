from pos_tagger.engines.EngineBase import EngineBase
from pos_tagger.engines.stanfordnlp_pos.StanfordNLPInst import \
    StanfordNLPInst


class StanfordNLPPOS(EngineBase):
    TYPE = 10
    NEEDS_GPU = True
    INST_CLASS = StanfordNLPInst

    def __init__(self, pos_taggers):
        EngineBase.__init__(self, pos_taggers)

    def download_engine(self, iso):
        # TODO!!!
        import stanfordnlp
        stanfordnlp.download(iso, force=False)

    def is_iso_supported(self, iso):
        return iso in self.get_L_supported_isos()

    def get_L_supported_isos(self):
        from stanfordnlp.utils.resources import default_treebanks
        return list(default_treebanks.keys())


if __name__ == '__main__':
    from glob import glob
    import stanfordnlp
    from os.path import expanduser, exists

    for iso in StanfordNLPPOS.get_L_supported_isos(None):
        if glob(expanduser(f'~/stanfordnlp_resources/{iso}_*models')):
            print("SKIPPING:", iso)
            continue

        print("DOWNLOADING:", iso)
        stanfordnlp.download(
            iso, force=True, confirm_if_exists=True
        )
