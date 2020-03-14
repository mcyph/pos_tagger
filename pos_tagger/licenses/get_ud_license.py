from collections import namedtuple


_UDLicense = namedtuple('UDLicense', [
    'model_name', 'license', 'iso'
])
LATEST_VERSION = '2.5'


def _get_D_ud_licenses(ud_version):
    # TODO: WHAT TO DO ABOUT CHINESE TRADITIONAL!?
    D = {}

    if ud_version == '2.5':
        from pos_tagger.licenses.ud_licenses_2_5 import ud_licenses_2_5 as ud_licenses
    elif ud_version == '2.4':
        from pos_tagger.licenses.ud_licenses_2_4 import ud_licenses_2_4 as ud_licenses
    elif ud_version == '2.3':
        from pos_tagger.licenses.ud_licenses_2_3 import ud_licenses_2_3 as ud_licenses
    elif ud_version == '2.2':
        from pos_tagger.licenses.ud_licenses_2_2 import ud_licenses_2_2 as ud_licenses
    else:
        raise Exception("Unsupported version: %s" % ud_version)

    for line in ud_licenses.strip().split('\n'):
        model_name, license = line.split('\t')
        model_name = model_name.strip()
        license = license.strip()
        iso = _model_to_iso(model_name)
        if iso is not None:
            D[model_name] = _UDLicense(model_name, license, iso)
    return D


iso_639_choices = [
    ('ab', 'Abkhaz'),
    ('aa', 'Afar'),
    ('af', 'Afrikaans'),
    ('ak', 'Akan'),
    ('sq', 'Albanian'),
    ('am', 'Amharic'),
    ('ar', 'Arabic'),
    ('an', 'Aragonese'),
    ('hy', 'Armenian'),
    ('as', 'Assamese'),
    ('av', 'Avaric'),
    ('ae', 'Avestan'),
    ('ay', 'Aymara'),
    ('az', 'Azerbaijani'),
    ('bm', 'Bambara'),
    ('ba', 'Bashkir'),
    ('eu', 'Basque'),
    ('be', 'Belarusian'),
    ('bn', 'Bengali'),
    ('bh', 'Bihari'),
    ('bi', 'Bislama'),
    ('bs', 'Bosnian'),
    ('br', 'Breton'),
    ('bg', 'Bulgarian'),
    ('my', 'Burmese'),
    ('ca', 'Catalan'),
    ('ch', 'Chamorro'),
    ('ce', 'Chechen'),
    ('ny', 'Chichewa; Chewa; Nyanja'),
    ('zh', 'Chinese'),
    ('cv', 'Chuvash'),
    ('kw', 'Cornish'),
    ('co', 'Corsican'),
    ('cr', 'Cree'),
    ('hr', 'Croatian'),
    ('cs', 'Czech'),
    ('da', 'Danish'),
    ('dv', 'Divehi'),
    ('nl', 'Dutch'),
    ('dz', 'Dzongkha'),
    ('en', 'English'),
    ('eo', 'Esperanto'),
    ('et', 'Estonian'),
    ('ee', 'Ewe'),
    ('fo', 'Faroese'),
    ('fj', 'Fijian'),
    ('fi', 'Finnish'),
    ('fr', 'French'),
    ('ff', 'Fula'),
    ('gl', 'Galician'),
    ('ka', 'Georgian'),
    ('de', 'German'),
    ('el', 'Greek'),
    ('gn', 'Guaraní'),
    ('gu', 'Gujarati'),
    ('ht', 'Haitian'),
    ('ha', 'Hausa'),
    ('he', 'Hebrew'),
    ('hz', 'Herero'),
    ('hi', 'Hindi'),
    ('ho', 'Hiri Motu'),
    ('hu', 'Hungarian'),
    ('ia', 'Interlingua'),
    ('id', 'Indonesian'),
    ('ie', 'Interlingue'),
    ('ga', 'Irish'),
    ('ig', 'Igbo'),
    ('ik', 'Inupiaq'),
    ('io', 'Ido'),
    ('is', 'Icelandic'),
    ('it', 'Italian'),
    ('iu', 'Inuktitut'),
    ('ja', 'Japanese'),
    ('jv', 'Javanese'),
    ('kl', 'Kalaallisut'),
    ('kn', 'Kannada'),
    ('kr', 'Kanuri'),
    ('ks', 'Kashmiri'),
    ('kk', 'Kazakh'),
    ('km', 'Khmer'),
    ('ki', 'Kikuyu, Gikuyu'),
    ('rw', 'Kinyarwanda'),
    ('ky', 'Kirghiz, Kyrgyz'),
    ('kv', 'Komi'),
    ('kg', 'Kongo'),
    ('ko', 'Korean'),
    ('ku', 'Kurdish'),
    ('kj', 'Kwanyama, Kuanyama'),
    ('la', 'Latin'),
    ('lb', 'Luxembourgish'),
    ('lg', 'Luganda'),
    ('li', 'Limburgish'),
    ('ln', 'Lingala'),
    ('lo', 'Lao'),
    ('lt', 'Lithuanian'),
    ('lu', 'Luba-Katanga'),
    ('lv', 'Latvian'),
    ('gv', 'Manx'),
    ('mk', 'Macedonian'),
    ('mg', 'Malagasy'),
    ('ms', 'Malay'),
    ('ml', 'Malayalam'),
    ('mt', 'Maltese'),
    ('mi', 'Māori'),
    ('mr', 'Marathi'),
    ('mh', 'Marshallese'),
    ('mn', 'Mongolian'),
    ('na', 'Nauru'),
    ('nv', 'Navajo, Navaho'),
    ('nb', 'Norwegian Bokmål'),
    ('nd', 'North Ndebele'),
    ('ne', 'Nepali'),
    ('ng', 'Ndonga'),
    ('nn', 'Norwegian Nynorsk'),
    ('no', 'Norwegian'),
    ('ii', 'Nuosu'),
    ('nr', 'South Ndebele'),
    ('oc', 'Occitan'),
    ('oj', 'Ojibwe, Ojibwa'),
    ('cu', 'Old Church Slavonic'),
    ('om', 'Oromo'),
    ('or', 'Oriya'),
    ('os', 'Ossetian, Ossetic'),
    ('pa', 'Panjabi'),
    ('pi', 'Pāli'),
    ('fa', 'Persian'),
    ('pl', 'Polish'),
    ('ps', 'Pashto'),
    ('pt', 'Portuguese'),
    ('qu', 'Quechua'),
    ('rm', 'Romansh'),
    ('rn', 'Kirundi'),
    ('ro', 'Romanian'),
    ('ru', 'Russian'),
    ('sa', 'Sanskrit'),
    ('sc', 'Sardinian'),
    ('sd', 'Sindhi'),
    ('se', 'Northern Sami'),
    ('sm', 'Samoan'),
    ('sg', 'Sango'),
    ('sr', 'Serbian'),
    ('gd', 'Scottish Gaelic'),
    ('sn', 'Shona'),
    ('si', 'Sinhala'),
    ('sk', 'Slovak'),
    ('sl', 'Slovene'),
    ('so', 'Somali'),
    ('st', 'Southern Sotho'),
    ('es', 'Spanish'),
    ('su', 'Sundanese'),
    ('sw', 'Swahili'),
    ('ss', 'Swati'),
    ('sv', 'Swedish'),
    ('ta', 'Tamil'),
    ('te', 'Telugu'),
    ('tg', 'Tajik'),
    ('th', 'Thai'),
    ('ti', 'Tigrinya'),
    ('bo', 'Tibetan'),
    ('tk', 'Turkmen'),
    ('tl', 'Tagalog'),
    ('tn', 'Tswana'),
    ('to', 'Tonga'),
    ('tr', 'Turkish'),
    ('ts', 'Tsonga'),
    ('tt', 'Tatar'),
    ('tw', 'Twi'),
    ('ty', 'Tahitian'),
    ('ug', 'Uyghur'),
    ('uk', 'Ukrainian'),
    ('ur', 'Urdu'),
    ('uz', 'Uzbek'),
    ('ve', 'Venda'),
    ('vi', 'Vietnamese'),
    ('vo', 'Volapük'),
    ('wa', 'Walloon'),
    ('cy', 'Welsh'),
    ('wo', 'Wolof'),
    ('fy', 'Western Frisian'),
    ('xh', 'Xhosa'),
    ('yi', 'Yiddish'),
    ('yo', 'Yoruba'),
    ('za', 'Zhuang, Chuang'),
    ('zu', 'Zulu'),

    ('akk', 'Akkadian'),
    ('grc', 'Ancient Greek'),
    ('aii', 'Assyrian'),
    ('bho', 'Bhojpuri'),
    ('bua', 'Buryat'),
    ('yue', 'Cantonese'),
    ('ltc', 'Classical Chinese'),
    ('cop', 'Coptic'),
    ('myv', 'Erzya'),
    ('got', 'Gothic'),
    (None, 'Hindi English'),
    ('krl', 'Karelian'),
    ('kur', 'Kurmanji'),
    ('olo', 'Livvi'),
    ('gun', 'Mbya Guarani'),
    ('mdf', 'Moksha'),
    ('ig', 'Naija'),
    ('sme', 'North Sami'),
    ('fro', 'Old French'),
    ('orv', 'Old Russian'),
    ('sms', 'Skolt'),
    ('sl', 'Slovenian'),
    ('swl', 'Swedish Sign Language'),
    ('gsw', 'Swiss German'),
    ('hsb', 'Upper Sorbian'),
    ('wbp', 'Warlpiri'),
    ('sms', 'Skolt Sami'),
    (None, 'Komi Zyrian'),
    (None, 'Komi Permyak'),
]

_DPreferredCommercial = {
    'zh': 'Chinese-GSDSimp',
    'zh_Hant': 'Chinese-GSD',
    'lt': 'Lithuanian-ALKSNIS',
    'ja': 'Japanese-GSD',
    'nl': 'Dutch-Alpino',
    'sv': 'Swedish-Talbanken',
    'id': 'Indonesian-GSD',
    'ko': 'Korean-Kaist',
    'en': 'English-EWT', # Note the models differ quite a lot!
    'fr': 'French-GSD',
    # Seems there's an exception when Romanian-RRT is used:
    # ValueError: [E167] Unknown morphological feature:
    # 'Case' (8245304235865630608).
    # This can happen if the tagger was trained with a different set of
    # morphological features. If you're using a pretrained model,
    # make sure that your models are up to date:
    # python -m spacy validate
    # I've changed to "Nonstandard" for now which includes the Bible and
    # poetry as bases, not sure how well it works on general texts
    'ro': 'Romanian-Nonstandard', #'Romanian-RRT',
    'es': 'Spanish-GSD',
    'no': 'Norwegian-Nynorsk', # ???
    'nn': 'Norwegian-Nynorsk',
    'nb': 'Norwegian-Bokmaal',
    'cz': 'Czech-CAC',
    'de': 'German-HDT',  # CHECK ME!
    'tr': 'Turkish-IMST',
    'pt': 'Portuguese-GSD',
    'fi': 'Finnish-TDT',
    'ar': 'Arabic-NYUAD',
    'ru': 'Russian-GSD',
}


def _model_to_iso(model_name):
    lang_name = model_name.split('-')[0]
    for iso, i_lang_name in iso_639_choices:
        if i_lang_name.lower() == lang_name.lower().replace('_', ' '):
            return iso
    print("ISO NOT FOUND:", model_name)
    return None


_DUDLicenses = {
    '2.5': _get_D_ud_licenses('2.5'),
    '2.4': _get_D_ud_licenses('2.4'),
    '2.3': _get_D_ud_licenses('2.3'),
    '2.2': _get_D_ud_licenses('2.2')
}


def get_L_ud_licenses(iso=None,
                      include_non_commercial=False,
                      include_gpl=False,
                      version=LATEST_VERSION):
    L = []
    for model_name, ud_license in _DUDLicenses[version].items():
        if iso == 'zh_Hant' and ud_license.iso == 'zh':
            pass  # allow for traditional Chinese, as UD doesn't differentiate
        elif iso is not None and ud_license.iso != iso:
            continue

        if 'NC' in ud_license.license and not include_non_commercial:
            continue
        elif 'GPL' in ud_license.license and not include_gpl:
            continue
        L.append(ud_license)

    if iso is not None:
        # Show preferred commercial ones first
        for ud_license in L:
            if (
                iso in _DPreferredCommercial and
                ud_license.model_name == _DPreferredCommercial[iso]
            ):
                item = L.pop(L.index(ud_license))
                L.insert(0, item)
    return L


def get_L_ud_iso_codes(include_non_commercial=False,
                       include_gpl=False,
                       version=LATEST_VERSION):
    S = set()
    for model_name, ud_license in _DUDLicenses[version].items():
        if 'NC' in ud_license.license and not include_non_commercial:
            continue
        elif 'GPL' in ud_license.license and not include_gpl:
            continue
        S.add(ud_license.iso)
        if ud_license.iso == 'zh':
            S.add('zh_Hant')
    return list(S)


def get_ud_license(ud_version, name):
    return _DUDLicenses[ud_version][name]


# Quick check there weren't errors
for __val in _DPreferredCommercial.values():
    get_ud_license('2.5', __val)
del __val


if __name__ == '__main__':
    print(get_L_ud_iso_codes())

    for iso in get_L_ud_iso_codes():
        if len(get_L_ud_licenses(iso)) > 1:
            print(get_L_ud_licenses(iso))

    print(get_L_ud_licenses('zh_Hant'))
    print(get_L_ud_licenses('zh'))
