import json
from glob import glob
from os import system, makedirs
from os.path import exists, expanduser
from pos_tagger.engines.spacy_pos.SpacyPOSBase import SpacyPOSBase
from pos_tagger.licenses.get_ud_license import get_L_ud_iso_codes, get_L_ud_licenses


MODELS_DIR = expanduser("~/.pos_tagger/spacy_ud")
if not exists(MODELS_DIR):
    makedirs(MODELS_DIR)

INCLUDE_NON_COMMERCIAL = False
INCLUDE_GPL = False


class SpacyUDPOS(SpacyPOSBase):
    """
    A subclass that downloads from git and trains from the
    Universal Dependencies archive.

    The advantage of doing this is, you'll get the latest
    versions of each of the engines, and non-commercial
    engines can be removed.
    """
    TYPE = 7

    def get_L_supported_isos(self):
        return get_L_ud_iso_codes(
            include_non_commercial=INCLUDE_NON_COMMERCIAL,
            include_gpl=INCLUDE_GPL
        )

    def _download_engine(self, iso):
        model_name = get_L_ud_licenses(
            iso=iso,
            include_non_commercial=INCLUDE_NON_COMMERCIAL,
            include_gpl=INCLUDE_GPL
        )[0].model_name

        # If engine already downloaded+trained, don't do it again!
        engine_path = f'{MODELS_DIR}/models/{model_name}/model-best'
        if exists(engine_path):
            return

        # Make the output dirs
        if not exists(f'{MODELS_DIR}/json_out'):
            makedirs(f'{MODELS_DIR}/json_out')
        if not exists(f'{MODELS_DIR}/models/{model_name}'):
            makedirs(f'{MODELS_DIR}/models/{model_name}')

        # Download the engine files from GitHub
        git_repo_path = f"{MODELS_DIR}/UD_{model_name}"
        if not exists(git_repo_path):
            system(f"git clone "
                   f"https://github.com/UniversalDependencies/UD_{model_name} "
                   f"{git_repo_path}")

        try:
            # Get the train/dev filenames
            conllu_train_fnam, conllu_dev_fnam = self.__get_train_dev_fnams(
                git_repo_path
            )
            # Convert them to json
            self.__convert_conllu_to_json(model_name, conllu_train_fnam)
            self.__convert_conllu_to_json(model_name, conllu_dev_fnam)
        except IndexError:
            conllu_train_fnam, conllu_dev_fnam = \
                self.__split_test_into_train_dev(git_repo_path, model_name)

        # Train the model
        try:
            exec(f"from spacy.lang import {iso}")
        except ImportError:
            # Probably not supported by Spacy!
            iso = 'xx'

        system(f"python3 -m spacy train {iso} {MODELS_DIR}/models/{model_name} "
               f"{MODELS_DIR}/json_out/"
                   f"{conllu_train_fnam.replace('.conllu', '.json')} "
               f"{MODELS_DIR}/json_out/"
                   f"{conllu_dev_fnam.replace('.conllu', '.json')}"
               f"{' --use-gpu 0' if self.use_gpu else ''}")

    def __convert_conllu_to_json(self, model_name, conllu_fnam):
        system(f"python3 -m spacy convert "
               f"{MODELS_DIR}/UD_{model_name}/{conllu_fnam} "
               f"{MODELS_DIR}/json_out")

    def __split_test_into_train_dev(self, git_repo_path, model_name):
        """
        Some UD repos only have a *-test.conllu file,
        so break it up into train/dev sets

        (ideally should have train/dev/test,
         not sure how to do that with spacy right now)
        """
        conllu_test_fnam = glob(
            f'{git_repo_path}/*-test.conllu'
        )[0].split('/')[-1]
        self.__convert_conllu_to_json(model_name, conllu_test_fnam)

        with open(
            f"{MODELS_DIR}/json_out/"
            f"{conllu_test_fnam.replace('.conllu', '.json')}",
            'r', encoding='utf-8'
        ) as f:
            LTest = json.loads(f.read())

        num_train = int(len(LTest)*0.8)
        LTrain = LTest[:num_train]
        LDev = LTest[num_train:]

        conllu_train_fnam = conllu_test_fnam.replace('.conllu', '.json').replace('test', 'train')
        conllu_dev_fnam = conllu_test_fnam.replace('.conllu', '.json').replace('test', 'dev')

        with open(
            f"{MODELS_DIR}/json_out/{conllu_train_fnam}",
            'w', encoding='utf-8'
        ) as f:
            f.write(json.dumps(LTrain))

        with open(
            f"{MODELS_DIR}/json_out/{conllu_dev_fnam}",
            'w', encoding='utf-8'
        ) as f:
            f.write(json.dumps(LDev))

        return conllu_train_fnam, conllu_dev_fnam

    def __get_train_dev_fnams(self, git_repo_path):
        conllu_train_fnam = glob(
            f'{git_repo_path}/*-train.conllu'
        )[0].split('/')[-1]
        conllu_dev_fnam = glob(
            f'{git_repo_path}/*-dev.conllu'
        )[0].split('/')[-1]
        return conllu_train_fnam, conllu_dev_fnam

    def _get_model(self, iso):
        import spacy
        model_name = get_L_ud_licenses(
            iso=iso,
            include_non_commercial=INCLUDE_NON_COMMERCIAL,
            include_gpl=INCLUDE_GPL
        )[0].model_name
        engine_path = f'{MODELS_DIR}/models/{model_name}/model-best'
        return spacy.load(engine_path)


if __name__ == '__main__':
    import fcntl
    SpacyUDPOS.use_gpu = True  # HACK!
    sud_pos = SpacyUDPOS(SpacyUDPOS)

    for iso in get_L_ud_iso_codes():
        # TODO: Make these locks work on Windows, too!
        lock_filename = '/tmp/spacy_ud_pos_train_%s.lock' % iso
        lock_file = open(lock_filename, 'w')
        try:
            fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print('Cannot lock: ' + lock_filename)
            continue

        try:
            sud_pos._download_engine(iso)
            sud_pos._get_model(iso)
        except:
            import traceback
            traceback.print_exc()
