The handwritten/printed classifier in this folder is the existing ConvNeXt-Tiny
production code from handwritten_printed_classifier. It is not a new model.

Source of truth:
  hw_printed.py
  models/handwritten_printed_convnext_tiny.pth

The pipeline calls it only through existing_classifier_adapter.py.
Do not reimplement or retrain it here.
