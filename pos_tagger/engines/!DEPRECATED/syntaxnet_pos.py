from urllib.request import urlretrieve

# TODO: Support either https://github.com/short-edition/syntaxnet-wrapper
# or https://github.com/JoshData/parsey-mcparseface-server
# or ...

for line in """
Ancient_Greek-PROIEL
Ancient_Greek
Arabic
Basque
Bulgarian
Catalan
Chinese
Croatian
Czech-CAC
Czech-CLTT
Czech
Danish
Dutch-LassySmall
Dutch
English-LinES
English
Estonian
Finnish-FTB
Finnish
French
Galician
German
Gothic
Greek
Hebrew
Hindi
Hungarian
Indonesian
Irish
Italian
Kazakh
Latin-ITTB
Latin-PROIEL
Latin
Latvian
Norwegian
Old_Church_Slavonic
Persian
Polish
Portuguese-BR
Portuguese
Romanian
Russian-SynTagRus
Russian
Slovenian-SST
Slovenian
Spanish-AnCora
Spanish
Swedish-LinES
Swedish
Tamil
Turkish
""".split('\n'):
    line = line.strip()
    if not line:
        continue

    urlretrieve(
        'http://download.tensorflow.org/models/parsey_universal/%s.zip' % line,
        'models/%s' % line
    )
