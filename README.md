# odg-region-exporter

Export odg regions to multiple pdf.

Rather than manually exporting selections in ODG file to pdf, this script makes it easy to define regions and export them automatically.

# How to use it

Create a box inside the odg file with text in `*.pdf.box` string format.
The box defines the boundaries of the region you wish to export.
The text specifies the name of the file to be produced (scheme1.pdf.box will produce the file scheme1.pdf).

# Usage

`exemple/test.fodg` contains 2 regions. `export_subpdf.py` will produces 2 pdf (schem1.pdf and sched2.pdf) containing the content of the box.
```
 ┌───────────────────────────────┐
 │┌──────────────────┐           │
 ││schem1.pdf.box    │           │
 ││                  │           │
 ││   XX------XX+    │           │
 ││                  │           │
 │└──────────────────┘           │
 │              ┌───────────────┐│
 │              │schem2.pdf.box ││
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
$ python export_subpdf.py exemple/test.fodg
# Display the generated pdf
$ evince ./figures/schem1.pdf
$ evince ./figures/schem2.pdf
```

# Notes

* The input file must be a `.fodg` file (Flat XML ODF Drawing). This makes it possible to use a simple xml parser instead of complex ODF libraries.
* Some ODG objects may not be detected in `MATCH_RULES`. You just need to add them and implement the associated `ExtendedElement` class. Contributions are welcome.


