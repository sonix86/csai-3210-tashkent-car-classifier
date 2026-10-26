# Rights and third-party components

This repository does not grant a blanket licence to all source, models and data.
The owner has not selected a licence for original project code or Tashkent photos.
No MIT/Apache/CC0 label should be inferred from the absence of a LICENSE file.
Publishing project code and assigning its licence are separate decisions.

Wikimedia Commons images retain their per-file CC0/CC BY/CC BY-SA terms.
See `data_external/ATTRIBUTIONS.csv`, `SOURCES_AND_LICENSES.md` and the image
`.license.json` sidecars. Cropping and resizing are documented modifications.
ShareAlike images must retain the applicable licence when redistributed.

YOLO11n is a general COCO detector, not a brand classifier. Its weights are not
bundled in the prepared project release assets. Ultralytics software/models have
their own licensing obligations (AGPL-3.0 or applicable enterprise terms):
https://www.ultralytics.com/license
https://github.com/ultralytics/ultralytics/blob/main/LICENSE

PyTorch/torchvision, ImageNet initialization and all other dependencies keep
their upstream licences. Project-trained checkpoints were initialized from a
general ImageNet backbone, then trained locally; they are not brand models
downloaded from a third party. See experiment configs and source snapshots.
