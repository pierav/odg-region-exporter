import sys
from lxml import etree

fname = sys.argv[1]

tree = etree.parse(fname)
root = tree.getroot()

def fixup(x):
    value = float(x[:-2])
    vr = round(value, 1)
    return f'{vr}cm'

for n in tree.iter():
    # print(n.attrib)
    for k, v in n.items():
        if not 'urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0' in k:
            continue
        if v.endswith('cm'):
            print(k, v)
            n.attrib[k] = fixup(v)

tree.write(fname + ".2.fodg", xml_declaration=True, encoding='UTF-8')

