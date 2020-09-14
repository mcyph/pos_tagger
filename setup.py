"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from codecs import open
from os import path
from os.path import join

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pos_tagger',
    version='0.1.0',
    description='Provides various title lookup indexes (like spellchecking, starts with, etc), with support for different languages',
    long_description=long_description,
    url='https://github.com/mcyph/pos_tagger',
    author='David Morrissey',
    author_email='david.l.morrissey@gmail.com',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Text Processing :: Indexing',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
    ],

    keywords='indexing',
    packages=find_packages(),

    install_requires=[
        # Here as pyvi depends on them, and it's a headache to
        # execute the following lines on PyPy if they aren't here
        # Might need to manually execute
        # pypy -m pip install numpy==1.15.4
        # https://stackoverflow.com/questions/54215712/cannot-import-numpy-in-pypy3-installs-fine-with-pip
        'numpy',
        'scipy',

        'pyvi',
        'pythainlp', 'pandas',
        'nlpcube',
        'jieba',  # as spacy might need it
        'jieba-fast',
        'spacy',
        'spacy-udpipe',
        'stanfordnlp',
        'PyKomoran',
        'pymorphy2',

        'mecab-python3',
        'natto-py',
        'fugashi',
        'sudachipy', # ==0.4.5
        'pkuseg',
    ],

    package_data={
        # Include the phonetic map *.txt data
        '': ['*.txt'],
    },

    zip_safe=False
)
