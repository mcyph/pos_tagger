SISOs = set()

for line in """
af-1.0
grc-1.0
ar-1.0
hy-1.0

eu-1.0
eu-1.1

bg-1.0
bg-1.1

bxr-1.0

ca-1.0
ca-1.1

zh-1.0
zh-1.1

hr-1.0
hr-1.1

cs-1.0
cs-1.1

da-1.0
da-1.1

nl-1.0
nl-1.1

en-1.0
en-1.1

et-1.0
et-1.1

fi-1.0
fi-1.1

fr-1.0
fr-1.1

gl-1.0
gl-1.1

de-1.0
de-1.1

got-1.0

el-1.0
el-1.1

he-1.0
he-1.1

hi-1.0
hi-1.1

hu-1.0
hu-1.1

id-1.0
id-1.1

ga-1.0

it-1.0
it-1.1

ja-1.0
ja-1.1

kk-1.0

ko-1.0
ko-1.1

kmr-1.0

la-1.0
la-1.1

lv-1.0

sme-1.0

no_bokmaal-1.0
no_bokmaal-1.1
no_nynorsk-1.0
no_nynorsk-1.1

cu-1.0

fa-1.0
fa-1.1

pt-1.0
pt-1.1

ro-1.0
ro-1.1

ru-1.0
ru-1.1

sr-1.0
sr-1.1

sk-1.0
sk-1.1

sl-1.0
sl-1.1

es-1.0
es-1.1

sv-1.0
sv-1.1

tr-1.0
tr-1.1

uk-1.0
uk-1.1

hsb-1.0

ur-1.0
ur-1.1

ug-1.0
ug-1.1

vi-1.0
vi-1.1
""".split('\n'):
    line = line.strip().strip('10.-')
    if not line:
        continue
    SISOs.add(line)

from pprint import pprint
pprint(sorted((i, i) for i in SISOs))
