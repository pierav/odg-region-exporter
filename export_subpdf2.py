#!/usr/bin/env python3

import sys
import os
import re
import logging
import shutil
import argparse
import subprocess
from decimal import Decimal
from functools import reduce
from typing import Optional, List, Type
from lxml import etree
from lxml.etree import QName, _Element

# ---------- Constants ----------
PAGE_DRAW_NODE_PATH = '//office:body/office:drawing/draw:page'
PAGE_LAYOUT_PROP_PATH = '//office:automatic-styles/style:page-layout/style:page-layout-properties'

# ---------- Logging Setup ----------
def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

# ---------- Utility Functions ----------
def check_lodraw():
    """Check if 'lodraw' (LibreOffice Draw) is available in PATH."""
    if shutil.which("lodraw") is None:
        logging.error("Could not find 'lodraw' in PATH. Please install LibreOffice and ensure 'lodraw' is in your PATH.")
        sys.exit(1)

def run_lodraw_convert(odg: str, outdir: str):
    """Convert a .fodg file to .pdf using LibreOffice Draw."""
    cmd = ["lodraw", "--convert-to", "pdf", "--outdir", outdir, odg]
    logging.info(f"Converting {odg} to PDF...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"PDF conversion failed: {e}")
        raise

def find_unique_node(root: _Element, xpath: str) -> _Element:
    matches = root.xpath(xpath, namespaces=root.nsmap)
    if len(matches) != 1:
        raise ValueError(f"XPath '{xpath}' did not yield exactly one result. Found: {len(matches)}")
    return matches[0]

def cm2decimal(x: Optional[str]) -> Optional[Decimal]:
    if x is None:
        return None
    if not x.endswith('cm'):
        raise ValueError(f"Value '{x}' must end with 'cm'.")
    return Decimal(x[:-2])

def decimal2cm(x: Decimal) -> str:
    return f"{x}cm"

def cm_minus_cm(cm1: str, cm2: str) -> str:
    return decimal2cm(cm2decimal(cm1) - cm2decimal(cm2))

def get_local_attr(elem: _Element, attr_name: str) -> Optional[str]:
    for full_name, value in elem.attrib.items():
        if QName(full_name).localname == attr_name:
            return value
    return None

def set_local_attr(elem: _Element, attr_name: str, value: str):
    for full_name in elem.attrib:
        if QName(full_name).localname == attr_name:
            elem.attrib[full_name] = value
            return
    # If the attribute is not present, add it with the default namespace
    # Use the same namespace as any existing attribute, otherwise fallback to no namespace
    for full_name in elem.attrib:
        ns = QName(full_name).namespace
        elem.attrib[f'{{{ns}}}{attr_name}'] = value
        return
    elem.attrib[attr_name] = value

# ---------- Bounding Box ----------
class BoundingBox:
    def __init__(self, x: Decimal, y: Decimal, w: Decimal, h: Decimal):
        self.x = x
        self.y = y
        self.xmax = x + w
        self.ymax = y + h

    @classmethod
    def from_boxes(cls, boxes: List['BoundingBox']) -> 'BoundingBox':
        if not boxes:
            raise ValueError("No bounding boxes to merge.")
        x = min(bb.x for bb in boxes)
        y = min(bb.y for bb in boxes)
        xmax = max(bb.xmax for bb in boxes)
        ymax = max(bb.ymax for bb in boxes)
        return cls(x, y, xmax - x, ymax - y)

    def contains_point(self, x: Decimal, y: Decimal) -> bool:
        return self.x <= x <= self.xmax and self.y <= y <= self.ymax

    def collides_with(self, bb: Optional['BoundingBox']) -> bool:
        if bb is None:
            return False
        # Only checks top-left point
        return self.contains_point(bb.x, bb.y)

    def width(self) -> Decimal:
        return self.xmax - self.x

    def height(self) -> Decimal:
        return self.ymax - self.y

    def to_cm_dict(self):
        return {
            'x': decimal2cm(self.x),
            'y': decimal2cm(self.y),
            'w': decimal2cm(self.width()),
            'h': decimal2cm(self.height())
        }

    def __repr__(self):
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.width()}, h={self.height()})"

# ---------- Element Adapters ----------
class ElementAdapter:
    @classmethod
    def get_bb(cls, elem: _Element) -> Optional[BoundingBox]:
        raise NotImplementedError

    @classmethod
    def apply_move(cls, elem: _Element, dx: str, dy: str):
        raise NotImplementedError

class GroupAdapter(ElementAdapter):
    @classmethod
    def get_bb(cls, elem: _Element) -> Optional[BoundingBox]:
        boxes = [get_element_bb(child) for child in elem if get_element_bb(child) is not None]
        if not boxes:
            return None
        return BoundingBox.from_boxes(boxes)

    @classmethod
    def apply_move(cls, elem: _Element, dx: str, dy: str):
        for child in elem:
            apply_element_move(child, dx, dy)

class IgnoreAdapter(ElementAdapter):
    @classmethod
    def get_bb(cls, elem: _Element) -> Optional[BoundingBox]:
        return None

    @classmethod
    def apply_move(cls, elem: _Element, dx: str, dy: str):
        pass

class LineAdapter(ElementAdapter):
    @classmethod
    def get_bb(cls, elem: _Element) -> Optional[BoundingBox]:
        d = {k: cm2decimal(get_local_attr(elem, k)) for k in ("x1", "y1", "x2", "y2")}
        if None in d.values():
            return None
        return BoundingBox(d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"])

    @classmethod
    def apply_move(cls, elem: _Element, dx: str, dy: str):
        for k in ("x1", "y1", "x2", "y2"):
            v = get_local_attr(elem, k)
            if v is not None:
                set_local_attr(elem, k, cm_minus_cm(v, dx if "x" in k else dy))

class XYWHAdapter(ElementAdapter):
    @classmethod
    def get_bb(cls, elem: _Element) -> Optional[BoundingBox]:
        d = {k: cm2decimal(get_local_attr(elem, k)) for k in ("x", "y", "width", "height")}
        if None in d.values():
            return None
        return BoundingBox(d["x"], d["y"], d["width"], d["height"])

    @classmethod
    def apply_move(cls, elem: _Element, dx: str, dy: str):
        for k, delta in [("x", dx), ("y", dy)]:
            v = get_local_attr(elem, k)
            if v is not None:
                set_local_attr(elem, k, cm_minus_cm(v, delta))

class TransformWHAdapter(ElementAdapter):
    @classmethod
    def parse_transform(cls, s: str):
        regex = r"rotate\s*\((-?\d+\.?\d*)\)\s*translate\s*\((-?\d+\.?\d*cm)\s+(-?\d+\.?\d*cm)\)"
        match = re.search(regex, s)
        if not match:
            raise ValueError(f"Invalid transform format: {s}")
        return match.group(1), match.group(2), match.group(3)

    @classmethod
    def write_transform(cls, rot: str, x: str, y: str):
        return f"rotate ({rot}) translate ({x} {y})"

    @classmethod
    def get_bb(cls, elem: _Element) -> Optional[BoundingBox]:
        transform = get_local_attr(elem, "transform")
        width = get_local_attr(elem, "width")
        height = get_local_attr(elem, "height")
        if None in (transform, width, height):
            return None
        rot, x, y = cls.parse_transform(transform)
        return BoundingBox(cm2decimal(x), cm2decimal(y), cm2decimal(width), cm2decimal(height))

    @classmethod
    def apply_move(cls, elem: _Element, dx: str, dy: str):
        transform = get_local_attr(elem, "transform")
        if not transform:
            return
        rot, x, y = cls.parse_transform(transform)
        new_x = cm_minus_cm(x, dx)
        new_y = cm_minus_cm(y, dy)
        set_local_attr(elem, "transform", cls.write_transform(rot, new_x, new_y))

MATCH_RULES = [
    {"name_regex": r"g", "keys": [], "adapter": GroupAdapter},
    {"name_regex": r"forms", "keys": [], "adapter": IgnoreAdapter},
    {"name_regex": r"line", "keys": ["x1", "y1", "x2", "y2"], "adapter": LineAdapter},
    {"name_regex": r".*", "keys": ["x", "y", "width", "height"], "adapter": XYWHAdapter},
    {"name_regex": r".*", "keys": ["transform", "width", "height"], "adapter": TransformWHAdapter},
]

def get_element_adapter(elem: _Element) -> Type[ElementAdapter]:
    name = QName(elem.tag).localname
    for rule in MATCH_RULES:
        if re.match(rule["name_regex"], name) and all(get_local_attr(elem, k) is not None for k in rule["keys"]):
            return rule["adapter"]
    raise ValueError(f"Unrecognized or unsupported element shape: {name}")

def get_element_bb(elem: _Element) -> Optional[BoundingBox]:
    return get_element_adapter(elem).get_bb(elem)

def apply_element_move(elem: _Element, dx: str, dy: str):
    get_element_adapter(elem).apply_move(elem, dx, dy)

def get_parent_custom_shape(elem: _Element) -> _Element:
    if QName(elem.tag).localname == "custom-shape":
        return elem
    parent = elem.getparent()
    if parent is None:
        raise ValueError("Could not find parent custom-shape element.")
    return get_parent_custom_shape(parent)

# ---------- Region ----------
class Region:
    def __init__(self, name: str, bbox: BoundingBox, args):
        self.box_name = name
        self.region_name = name[:-8]  # Remove '.pdf.box'
        self.bbox = bbox
        self.args = args

    def output_path(self, suffix: str = "") -> str:
        return os.path.join(self.args.output, self.region_name + suffix)

    @classmethod
    def extract_regions(cls, fname: str, root: _Element, args) -> List['Region']:
        regions = []
        for tag in root.iter():
            if tag.text and tag.text.endswith(".pdf.box"):
                bbox = get_element_bb(get_parent_custom_shape(tag))
                regions.append(cls(tag.text, bbox, args))
        return regions

    def generate_pdf(self, src_file: str):
        tree = etree.parse(src_file)
        root = tree.getroot()

        # Remove the region's box element
        box_elem = None
        for tag in root.iter():
            if tag.text and tag.text == self.box_name:
                box_elem = get_parent_custom_shape(tag)
                break
        if box_elem is None:
            raise ValueError(f"Box element for region '{self.box_name}' not found.")
        box_elem.getparent().remove(box_elem)

        # Pass 0: Filter elements not in bounding box
        page_draw = find_unique_node(root, PAGE_DRAW_NODE_PATH)
        for elem in list(page_draw):  # copy to allow removal
            if not self.bbox.collides_with(get_element_bb(elem)):
                page_draw.remove(elem)

        if self.args.dump_fodg:
            path = self.output_path('.0.fodg')
            tree.write(path, xml_declaration=True, encoding='UTF-8')
            logging.info(f"Wrote intermediate file: {path}")

        # Pass 1: Fix page size and margins
        if self.args.tight:
            box_dict = BoundingBox.from_boxes([get_element_bb(e) for e in page_draw if get_element_bb(e) is not None]).to_cm_dict()
        else:
            box_dict = get_element_bb(box_elem).to_cm_dict()

        page_layout = find_unique_node(root, PAGE_LAYOUT_PROP_PATH)
        fo_ns = f'{{{root.nsmap["fo"]}}}'
        for margin in ["margin-top", "margin-bottom", "margin-left", "margin-right"]:
            page_layout.attrib[f'{fo_ns}{margin}'] = '0cm'
        page_layout.attrib[f'{fo_ns}page-width'] = box_dict["w"]
        page_layout.attrib[f'{fo_ns}page-height'] = box_dict["h"]

        if self.args.dump_fodg:
            path = self.output_path('.1.fodg')
            tree.write(path, xml_declaration=True, encoding='UTF-8')
            logging.info(f"Wrote intermediate file: {path}")

        # Pass 2: Shift everything to the new origin
        for elem in page_draw:
            apply_element_move(elem, box_dict["x"], box_dict["y"])
            bb = get_element_bb(elem)
            if bb is not None:
                assert bb.x >= 0 and bb.y >= 0, "Element outside positive region"
                assert bb.xmax <= cm2decimal(box_dict["w"])
                assert bb.ymax <= cm2decimal(box_dict["h"])

        fodg_path = self.output_path('.fodg')
        tree.write(fodg_path, xml_declaration=True, encoding='UTF-8')
        logging.info(f"Region '{self.region_name}': wrote FODG to {fodg_path}")

        # Conversion to PDF
        run_lodraw_convert(fodg_path, self.args.output)
        pdf_path = self.output_path('.pdf')
        if os.path.exists(pdf_path):
            logging.info(f"Region '{self.region_name}': PDF exported at {pdf_path}")
        else:
            logging.warning(f"Region '{self.region_name}': PDF not found after export.")

        if not self.args.dump_fodg:
            try:
                os.remove(fodg_path)
            except Exception as e:
                logging.warning(f"Could not remove temp FODG: {fodg_path}: {e}")

# ---------- CLI and Main ----------
def parse_args():
    parser = argparse.ArgumentParser(
        prog="export_subpdf.py",
        description="Export ODG regions to multiple PDFs.",
        epilog=":)"
    )
    parser.add_argument("file", help="The .fodg file to process.")
    parser.add_argument("-o", "--output", default="./figures", help="Output directory (default: ./figures)")
    parser.add_argument("--tight", action='store_true', help="Use tight bounding box (default: False)")
    parser.add_argument("--dump-fodg", action='store_true', help="Dump intermediate FODG files.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Enable verbose output.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    check_lodraw()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Parse the input FODG file
    try:
        tree = etree.parse(args.file)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"Could not parse input file '{args.file}': {e}")
        sys.exit(1)

    # Check that the structure is as expected
    try:
        find_unique_node(root, PAGE_DRAW_NODE_PATH)
        find_unique_node(root, PAGE_LAYOUT_PROP_PATH)
    except Exception as e:
        logging.error(f"Invalid FODG structure: {e}")
        sys.exit(1)

    # Extract regions and generate PDFs
    regions = Region.extract_regions(args.file, root, args)
    if not regions:
        logging.error("No regions found (no elements ending with '.pdf.box').")
        sys.exit(1)

    for region in regions:
        try:
            region.generate_pdf(args.file)
        except Exception as e:
            logging.error(f"Failed to export region '{region.region_name}': {e}")

if __name__ == "__main__":
    main()
