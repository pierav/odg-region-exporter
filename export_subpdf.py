#!/usr/bin/env python3

import sys
from lxml import etree
from lxml.etree import QName
from lxml.etree import _Element
from decimal import Decimal # Use decimal as main unit type
import subprocess # For odg to pdf
import argparse
import os.path
from functools import reduce
from operator import add
import re

# Some config 
TIGHT = True # By default une tight bounding box, not the drawn one
DUMP_FODG = False # Dump intermediate fodg files

# xml nodes to updates
PAGE_DRAW_NODE_PATH = '//office:body/office:drawing/draw:page'
PAGE_LAYOUT_PROP_PATH = '//office:automatic-styles/style:page-layout/style:page-layout-properties'


def odg2pdf(odg):
    outdir = os.path.dirname(odg)
    cmd = f"lodraw --convert-to pdf --outdir {outdir} {odg}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def findUniqueNode(root, path):
    # Find page content
    matchs = root.xpath(path, namespaces=root.nsmap)
    if len(matchs) > 1:
        raise Exception(f"Excedded number of nodes for path: {path}")
    if not len(matchs):
        raise Exception(f"Cannot find node for path: {path}")
    return matchs[0]

def cm2int(x: str) -> Decimal:
    if x is None:
        return None
    if x[-2:] != 'cm':
        raise Exception(f'{x} must be in cm')
    res = Decimal(x[:-2])
    return res

def int2cm(x: Decimal) -> str:
    return str(x) + 'cm'

def cmMinusCm(cm1: str, cm2: str) -> str:
    return int2cm(cm2int(cm1) - cm2int(cm2))

class BoundingBox:
    """
    An usual bounding box used to perform simple geometry
    """
    def __init__(self, x: float, y: float, width: float, height: float):
        """ Default constructor """
        self.x = x
        self.y = y
        self.xmax = x + width
        self.ymax = y + height


    @classmethod
    def fromlist(cls, it):
        """ Construct the englobing bounding box """
        return reduce(add, it)

    def isPointIn(self, x, y):
        return self.x <= x <= self.xmax and \
               self.y <= y <= self.ymax
    
    def isCollide(self, bb):
        if bb is None:
            return False
        return self.isPointIn(bb.x, bb.y) # False negative

    def __repr__(self):
        return f"BoundingBox({self.x}, {self.y}, {self.xmax}, {self.ymax})"

    def width(self) -> int:
        return self.xmax - self.x

    def height(self) -> int:
        return self.ymax - self.y

    def merge(self, other):
        self.x = min(self.x, other.x)
        self.y = min(self.y, other.y)
        self.xmax = max(self.xmax, other.xmax)
        self.ymax = max(self.ymax, other.ymax)
        return self

    def __add__(self, other):
        """Override add func to perform reduce add"""
        return self.merge(other)

    def toCmDict(self):
        return {'x': int2cm(self.x),
                'y': int2cm(self.y),
                'w': int2cm(self.width()),
                'h': int2cm(self.height())}


# Get the attribute by local name, without specifying the namespace
def getattr(elem, attr_name):
    for full_attr_name in elem.attrib:
        if etree.QName(full_attr_name).localname == attr_name:
            return elem.attrib[full_attr_name]
    return None


# Set the attribute by local name, without specifying the namespace
def setattr(elem, attr_name, value):
    for full_attr_name in elem.attrib:
        if etree.QName(full_attr_name).localname == attr_name:
            elem.attrib[full_attr_name] = value


class ExtendedElementXYXY:
    @classmethod
    def getBB(cls, e):
        NS = f'{{{e.nsmap["svg"]}}}'
        d = {k:cm2int(getattr(e, k)) for k in ('x1', 'y1', 'x2', 'y2')}
        assert(not None in d.values())
        return BoundingBox(d['x1'], d['y1'], d['x2'] - d['x1'], d['y2'] - d['y1'])

    @classmethod
    def applyMove(cls, e, dx, dy):
        d = {k:getattr(e, k) for k in ('x1', 'y1', 'x2', 'y2')}
        assert(not None in d.values())
        delta = lambda k: dx if 'x' in k else dy
        for k, v in d.items():
            setattr(e, k, cmMinusCm(v, delta(k)))


class ExtendedElementXYWH:
    @classmethod
    def getBB(cls, e):
        d = {k:cm2int(getattr(e, k)) for k in ('x', 'y', 'width', 'height')}
        assert(not None in d.values())
        return BoundingBox(d['x'], d['y'], d['width'], d['height'])

    @classmethod
    def applyMove(cls, e, dx, dy):
        setattr(e, 'x', cmMinusCm(getattr(e, 'x'), dx))
        setattr(e, 'y', cmMinusCm(getattr(e, 'y'), dy))


class ExtendedElementTransformWH:
    @classmethod
    def parseTransform(cls, s):
        regex = r'rotate\s*\((-?\d+\.?\d*)\)\s*translate\s*\((-?\d+\.?\d*cm)\s+(-?\d+\.?\d*cm)\)'
        # 'rotate (-1.40568817955623) translate (16.0004500426334cm 21.5188594413585cm)'
        match = re.search(regex, s)
        if not match:
            raise Exception(f"Invalid transform format: {s}")
        return match.group(1), match.group(2), match.group(3)
    
    @classmethod
    def writeTransform(cls, rot, x, y):
        return f"rotate ({rot}) translate ({x} {y})"

    @classmethod
    def getBB(cls, e):
        d = {k:getattr(e, k) for k in ('transform', 'width', 'height')}
        assert(not None in d.values())
        rot, x, y = cls.parseTransform(d['transform'])
        # print("TRANSFORM !!!", rot, x, y, d[NS+'width'], d[NS+'height'])        
        return BoundingBox(cm2int(x), cm2int(y), cm2int(d['width']), cm2int(d['height']))
        
    @classmethod
    def applyMove(cls, e, dx, dy):
        rot, x, y = cls.parseTransform(getattr(e, 'transform'))
        x = cmMinusCm(x, dx)
        y = cmMinusCm(y, dy)
        setattr(e, 'transform', cls.writeTransform(rot, x, y))


class ExtendedElementGroup:
    @classmethod
    def getBB(cls, e):
        return BoundingBox.fromlist(map(getBB, e))
        
    @classmethod
    def applyMove(cls, e, dx, dy):
        for sube in e:
            applyMove(sube, dx, dy)


class ExtendedElementIgnore:
    @classmethod
    def getBB(cls, e):
        return None # No BB

    @classmethod
    def applyMove(cls, e, dx, dy):
        pass # Nothing to do


# RULE : match name, match keys, class type
MATCH_RULES = [
    (r'g', (), ExtendedElementGroup),
    (r'forms', (), ExtendedElementIgnore),
    (r'line', ('x1', 'y1', 'x2', 'y2'), ExtendedElementXYXY),
    (r'.*', ('x', 'y', 'width', 'height'), ExtendedElementXYWH),
    (r'.*', ('transform', 'width', 'height'), ExtendedElementTransformWH),
]


def getMatchClass(e: _Element):
    name = QName(e.tag).localname
    for rr, keys, model in MATCH_RULES:
        if re.match(rr, name) and not None in (getattr(e, k) for k in keys):
            return model

    print(f'warning: {name} {e}')
    for kv in e.attrib.items():
            print(kv)
    
    raise Exception("Invalid shape")


def getBB(e: _Element) -> BoundingBox:
    return getMatchClass(e).getBB(e)


def applyMove(e: _Element, dx, dy):
    return getMatchClass(e).applyMove(e, dx, dy)


def getParentShapeElement(e: _Element) -> _Element:
    if QName(e.tag).localname == 'custom-shape':
        return e
    else:
        return getParentShapeElement(e.getparent())


class Region:
    """
    The main class used to apply all passes to an lodg file for each region
    """
    fname = None # The input file name of this region

    def __init__(self, name: str, bb: BoundingBox):
        print(f"Find region {name} : {bb} in {Region.fname}")
        assert(name.endswith(".pdf.box"))
        self.boxname = name
        self.name = name[:-8]
        self.bb = bb
        self.elems = []

    def isCollide(self, bb: BoundingBox) -> bool:
        return self.bb.isCollide(bb)

    def insertElem(self, e: _Element) -> None:
        self.elems.append(e)

    def __repr__(self) -> str:
        return f"Region({self.name}, {self.bb})"

    def getWritePath(self, suffix='') -> str:
        return args.o + '/' + self.name + suffix

    def generatePdf(self):
        # make a copy TODO; reread the file to perform inplace modifications
        print(f"Begin pdf generation for {self} in {Region.fname}...")
        tree = etree.parse(Region.fname)
        root = tree.getroot()

        # Quick test: find the box
        elemBox = None
        for tag in root.iter():
            if tag.text and self.boxname == tag.text:
                elemBox = getParentShapeElement(tag)
                break
        assert(elemBox is not None)

        # Remove it from the root
        elemBox.getparent().remove(elemBox)

        # Pass 0: Filter elementes that are not in the box
        basenode = findUniqueNode(root, PAGE_DRAW_NODE_PATH)
        for e in basenode:
            if not self.isCollide(getBB(e)):
                e.getparent().remove(e)

        if DUMP_FODG:
            wname = self.getWritePath('.0.fodg')
            print(f"PASS 0: Write file {wname}")
            tree.write(wname, xml_declaration=True, encoding='UTF-8')
        
        # Pass 1: fix page size
        # Compute tight bounding box
        if TIGHT:
            elemBoxDict = BoundingBox.fromlist(map(getBB, basenode)).toCmDict()
        else:
            elemBoxDict = getBB(elemBox).toCmDict()

        # print(f"Box BoundingBox is {elemBoxDict}")

        node = findUniqueNode(root, PAGE_LAYOUT_PROP_PATH)
        FO_NS = f'{{{root.nsmap["fo"]}}}'
        # print(etree.tostring(node).decode())
        # <style:page-layout-properties
        # fo:margin-top="1cm" fo:margin-bottom="1cm" fo:margin-left="1cm" fo:margin-right="1cm"
        # fo:page-width="21cm" fo:page-height="29.7cm" style:print-orientation="portrait"/>
        # Set 0 margin
        node.attrib[f'{FO_NS}margin-top'] = '0cm'
        node.attrib[f'{FO_NS}margin-bottom'] = '0cm'
        node.attrib[f'{FO_NS}margin-left'] = '0cm'
        node.attrib[f'{FO_NS}margin-right'] = '0cm'
        node.attrib[f'{FO_NS}page-width'] = elemBoxDict['w']
        node.attrib[f'{FO_NS}page-height'] = elemBoxDict['h']
        # print(etree.tostring(node).decode())

        if DUMP_FODG:
            wname = self.getWritePath('.1.fodg')
            print(f"PASS 1: Write file {wname}")
            tree.write(wname, xml_declaration=True, encoding='UTF-8')

        # Pass 2: Shift everything to the corner
        for e in basenode:
            applyMove(e, elemBoxDict['x'], elemBoxDict['y'])
            # Perfom check to test if the figure is in the page
            bb = getBB(e)
            assert(bb.x >= 0)
            assert(bb.y >= 0)
            assert(bb.xmax <= cm2int(elemBoxDict["w"]))
            assert(bb.ymax <= cm2int(elemBoxDict["h"]))

        # The last fodg before conversion to pdf
        wname = self.getWritePath('.fodg')
        if DUMP_FODG:
            print(f"PASS 2: Write file {wname}")
        tree.write(wname, xml_declaration=True, encoding='UTF-8')

        # Final : convert fodg to pdf
        odg2pdf(wname)

        # Clean if needed
        if not DUMP_FODG:
            os.remove(wname)

    @classmethod
    def initRegions(cls, fname, root):
        cls.fname = fname
        boxs = []
        for tag in root.iter():
            if tag.text and tag.text.endswith(".pdf.box"):
                # print(tag, tag.text)
                boxs.append(Region(tag.text, getBB(getParentShapeElement(tag))))
        
        return boxs


# Parse args
parser = argparse.ArgumentParser(
    prog='export_subpdf.py',
    description='Export odg regions to multiple pdf',
    epilog=':)')

parser.add_argument('file', help='The fodg file to process')
parser.add_argument('-o', default='./figures', help='output directory')

args = parser.parse_args()

# Sanitize args
if not os.path.isdir(args.o):
    raise Exception(f"{args.o} is not a valid path")

tree = etree.parse(args.file)
root = tree.getroot()
# Ensures early xml paths are valids
findUniqueNode(root, PAGE_DRAW_NODE_PATH)
findUniqueNode(root, PAGE_LAYOUT_PROP_PATH)

# Get regions
regions = Region.initRegions(args.file, root)

# Generate pdfs
for region in regions:
    region.generatePdf()


