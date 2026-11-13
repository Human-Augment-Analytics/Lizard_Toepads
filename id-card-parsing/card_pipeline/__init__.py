"""End-to-end ID card OCR pipeline.

YOLO-OBB detect -> canonicalize corners -> rectify/normalize -> template field
crop -> PaddleOCR with a per-field dictionary -> JSON + CSV. Checkbox regions
are cropped and recorded as deferred. Model-backed pieces (detector,
recognizer) sit behind protocols so everything else is unit-testable headlessly.
"""
