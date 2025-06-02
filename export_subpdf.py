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
import logging
from more_itertools import one
import math


# xml nodes to updates
PAGE_DRAW_NODE_PATH = '//office:body/office:drawing/draw:page'
PAGE_LAYOUT_PROP_PATH = '//office:automatic-styles/style:page-layout/style:page-layout-properties'


# Some IO file manipulations
def odg2pdf(odg):
    logging.info(f"Creating PDF: {odg.replace('.fodg', '.pdf')}")
    outdir = os.path.dirname(odg)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cmd = [
        "lodraw",
        "--convert-to", "pdf",
        "--outdir", str(outdir),
        odg
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"PDF conversion failed: {e}")
        raise


def writeOdg(tree, wname):
    logging.info(f"Create file {wname}")
    outdir = os.path.dirname(wname)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tree.write(wname, xml_declaration=True, encoding='UTF-8')


def findUniqueNode(root, path):
    # Find page content
    matchs = root.xpath(path, namespaces=root.nsmap)
    if len(matchs) > 1:
        raise Exception(f"Excedded number of nodes for path: {path}")
    if not len(matchs):
        raise Exception(f"Cannot find node for path: {path}")
    return matchs[0]


def textifyNode(e: _Element) -> str:
    if etree.QName(e).localname != 'p':
        return None
    return ''.join(x.text for x in (e, *e) if x.text)


def findNodesByText(root, regex):
    for e in root.iter():
        s = textifyNode(e)
        if s and re.match(regex, s):
            yield s, e

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


def rotated_bounding_box(x, y, w, h, ang):
    """Compute the axis-aligned bounding box of a rectangle after rotation.
    """
    # base point in polar cords
    w, h = w, h
    dts = [[0, 0], [w, 0], [h, math.pi/2], [math.sqrt(w**2 + h**2), math.atan(h / w)]]
    # Rotate in polar cords
    for dt in dts:
        dt[1] -= ang
    xs = [x + d * math.cos(t) for d, t in dts]
    ys = [y + d * math.sin(t) for d, t in dts]
    # New bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x - min_x, max_y - min_y


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
        # Apply the rotation to the box
        xywh = tuple(map(lambda x: float(cm2int(x)), (x, y, d['width'], d['height'])))
        box = BoundingBox(*map(Decimal, rotated_bounding_box(*xywh, float(rot))))
        return box

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

    logging.error(f'Invalid shape {name} {e}')
    for kv in e.attrib.items():
            logging.error(kv)
    
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
    args = None
    # The unique thee of this region
    tree = None

    def __init__(self, name: str, bb: BoundingBox):
        assert(name.endswith(".pdf.box"))
        self.boxname = name
        self.bb = bb
        self.name = name[:-8]

    def isCollide(self, bb: BoundingBox) -> bool:
        return self.bb.isCollide(bb)

    def __repr__(self) -> str:
        return f"Region({self.name}, {self.bb})"

    def getWritePath(self, suffix='') -> str:
        return Region.args.o + '/' + self.name + suffix

    def debugIntermediateResult(self, key=''):
        if Region.args.dumpfodg:
            writeOdg(self.tree, self.getWritePath(key + 'fodg'))  

    def mapToBasenode(self, f):
        basenode = findUniqueNode(root, PAGE_DRAW_NODE_PATH)
        return list(map(f, basenode))

    def generatePdf(self):
        # make a copy TODO; reread the file to perform inplace modifications
        logging.info(f"Begin pdf generation for {self} in {Region.args.file}...")
        self.tree = etree.parse(Region.args.file)
        root = self.tree.getroot()

        # Quick test: find the box
        _, boxnode = one(findNodesByText(root, self.boxname))
        boxnode = getParentShapeElement(boxnode)

        # Remove it from the root
        boxnode.getparent().remove(boxnode)

        # Pass 0: Filter elementes that are not in the box
        basenode = findUniqueNode(root, PAGE_DRAW_NODE_PATH)
        for e in basenode:
            if not self.isCollide(getBB(e)):
                e.getparent().remove(e)

        self.debugIntermediateResult('.0')
        
        # Pass 1: fix page size
        # Compute tight bounding box
        if Region.args.notight:
            box_cm = getBB(boxnode).toCmDict()
        else:
            box_cm = BoundingBox.fromlist(map(getBB, basenode)).toCmDict()
        
        # Set 0 margin
        node = findUniqueNode(root, PAGE_LAYOUT_PROP_PATH)
        FO_NS = f'{{{root.nsmap["fo"]}}}'
        for m in ['top', 'bottom', 'right', 'left']:
            node.attrib[f'{FO_NS}margin-{m}'] = '0cm'
        
        node.attrib[f'{FO_NS}page-width'] = box_cm['w']
        node.attrib[f'{FO_NS}page-height'] = box_cm['h']        
        self.debugIntermediateResult('.1')

        
        # Pass 2: Shift everything to the corner
        for e in basenode:
            applyMove(e, box_cm['x'], box_cm['y'])
            # Perfom check to test if the figure is in the page
            # bb = getBB(e)
            # assert(bb.x >= 0)
            # assert(bb.y >= 0)
            # assert(bb.xmax <= cm2int(box_cm["w"]))
            # assert(bb.ymax <= cm2int(box_cm["h"]))

        # The last fodg before conversion to pdf
        wname = self.getWritePath('.fodg')
        writeOdg(self.tree, wname)

        # Final : convert fodg to pdf
        odg2pdf(wname)

        # Clean if needed
        if not Region.args.dumpfodg:
            os.remove(wname)

    @classmethod
    def boxNameMatchPrefix(cls, s: str) -> bool:
        return os.path.dirname(s) == cls.args.prefixpath

    @classmethod
    def initRegions(cls, args, root):
        cls.args = args
        boxs = []
        for name, e in findNodesByText(root, r".*.pdf.box"):
            if cls.boxNameMatchPrefix(name):
                logging.info(f"Process region : {name}")
                boxs.append(Region(name, getBB(getParentShapeElement(e))))
            else:
                logging.info(f"Ignore region : {name}")
        return boxs


def getRectNode(node, x, y, w, h):
    ns = node.nsmap
    snode = etree.Element(f"{{{ns['draw']}}}custom-shape", nsmap=node.nsmap)
    NSSVG = f'{{{ns["svg"]}}}'
    NSDRAW = f'{{{ns["draw"]}}}'
    kv = {
        f'{{{ns["draw"]}}}style-name': "gr8",
        f'{{{ns["draw"]}}}text-style-name': "P2",
        f'{{{ns["draw"]}}}layer': "layout",
        f'{{{ns["svg"]}}}width': w,
        f'{{{ns["svg"]}}}height': h,
        f'{{{ns["svg"]}}}x': x,
        f'{{{ns["svg"]}}}y': y
    }
    for k, v in kv.items():
        snode.attrib[k] = v

    gnode = etree.Element(f"{{{ns['draw']}}}enhanced-geometry", nsmap=node.nsmap)
    gnode.attrib[NSSVG + 'viewBox'] = "0 0 21600 21600"
    gnode.attrib[NSDRAW + 'type'] = "rectangle"
    gnode.attrib[NSDRAW + 'enhanced-path'] = "M 0 0 L 21600 0 21600 21600 0 21600 0 0 Z N"

    snode.append(gnode)
    return snode


def node2bbnode(node):
    bb = getBB(node).toCmDict()
    return getRectNode(node, *(bb[x] for x in 'xywh'))


def main():
    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Parse args
    parser = argparse.ArgumentParser(
        prog='export_subpdf.py',
        description='Export odg regions to multiple pdf',
        epilog=':)')
    parser.add_argument('file', help='The fodg file to process')
    parser.add_argument('-o', default='./figures', help='output directory')
    parser.add_argument('--prefixpath', default='', help='prefix to not dump everything')
    parser.add_argument('--notight', action='store_true', help='Crop the boxes')
    parser.add_argument('--dumpfodg', action='store_true', help='Dump debug files')
    args = parser.parse_args()

    # Sanitize args
    tree = etree.parse(args.file)
    root = tree.getroot()
    # Ensures early xml paths are valids
    basenode = findUniqueNode(root, PAGE_DRAW_NODE_PATH)
    findUniqueNode(root, PAGE_LAYOUT_PROP_PATH)

    # Draw bounding box
    # newnodes = [node2bbnode(node) for node in basenode if getBB(node)]
    # for node in newnodes:
    #     basenode.append(node)
    # writeOdg(tree, args.o + '/box.fodg')
    # exit(1)

    # Get regions
    regions = Region.initRegions(args, root)

    # Generate pdfs
    for region in regions:
        region.generatePdf()


if __name__ == "__main__":
    main()