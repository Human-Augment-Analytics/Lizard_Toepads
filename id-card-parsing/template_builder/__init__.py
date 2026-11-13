"""Build a reusable field template from one blank ID card image.

Load a card, hand-crop it as an oriented box, rectify and resize-pad it onto a
configurable square canvas, then draw/name/type one bounding box per field and
save a single Template_Record JSON for the later OCR pipeline to consume.

The backend (config, normalize, template_record, export_layout, backend) has no
GUI dependency and is unit-tested headlessly. The ``ui`` subpackage is the thin
Tkinter layer.
"""
