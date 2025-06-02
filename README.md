# odg-region-exporter

**Export ODG regions to multiple PDFs**

This tool allows you to extract and export specific regions from [LibreOffice Draw](https://www.libreoffice.org/discover/draw/) `.odg` files (in FODG format) into separate PDF files. It is particularly useful for splitting schematic or annotated diagrams into multiple, precisely-cropped PDF outputs based on region markers inside the drawing.

# How to use it

1. **Mark regions to export:**
   In your ODG file (save as `.fodg`), draw a box around the content you want to export.
   Inside the box, add a text label with the format `[<path>/]<output-filename>.pdf.box`
   For example, a box labeled `scheme1.pdf.box` will export its content to `scheme1.pdf`.

2. **Example region layout:**

`example/test.fodg` contains 2 regions. `export_subpdf.py` will produces 2 PDFs (scheme1.pdf and scheme2.pdf) containing the content of the box.
```
 ┌───────────────────────────────┐
 │┌──────────────────┐           │
 ││scheme1.pdf.box   │           │
 ││                  │           │
 ││   XX------XX+    │           │
 ││                  │           │
 │└──────────────────┘           │
 │              ┌───────────────┐│
 │              │scheme2.pdf.box││
 │              │               ││
 │              │               ││
 │              │   XXXX        ││
 │              │       YYYYY   ││
 │              │         │     ││
 │              │         │     ││
 │              │   ZZZ───┘     ││
 │              │               ││
 │              │               ││
 │              └───────────────┘│
 └───────────────────────────────┘
```

```sh
# Generate one pdf per region
$ python export_subpdf.py example/test.fodg
```

Exported PDFs will be saved (by default) in the `figures/` directory:

```
# Display the generated pdf
$ evince ./figures/scheme1.pdf
$ evince ./figures/scheme2.pdf
```

# Notes

* The input file must be a `.fodg` file (Flat XML ODF Drawing). This makes it possible to use a simple xml parser instead of complex ODF libraries.
* Some ODG objects may not be detected in `MATCH_RULES`. You just need to add them and implement the associated `ExtendedElement` adapter class. Contributions are welcome.


