=====
About
=====

A multi-engine part-of-speech tagging system.

Brings together the following engines:

* Adobe cube NLP
* Jieba
* PyKomoran
* PyThaiNLP
* pyvi
* Spacy
* Stanford NLP

Not ready for general use - built for use at http://langlynx.com

============
Requirements
============

TODO!

apt install natto mecab-ko mecab-ko-dic
natto-py

============
Installation
============

TODO!

============
TODO
============

* Provide testing to make sure things are setup correctly, turning off engines which aren't functioning (machine learning setups can be complex, and components can conflict or be difficult to setup, so better partially working than not at all!)
* Allow downloading of models (either on-demand or explicitly).
* Add support for MeCab (Japanese)
* Add support for selecting models by license (LGPL/CC-BY/CC-BY-NC) etc
* Add serialization/deserialization with "pretty-printed" HTML in a (somewhat) standard format, allowing for doing things like javascript identification on mouseover of dependencies, and the lemmatized forms of words

=====================
Bugs/Feature Requests
=====================

Please report any bugs/feature requests at GitHub:
https://github.com/mcyph/pos_tagger
