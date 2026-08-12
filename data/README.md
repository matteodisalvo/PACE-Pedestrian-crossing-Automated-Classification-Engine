# Data

- `raw/subset-IPA-AIA-crossing/` — subset of the Mapillary Traffic Sign dataset: 155 street-level images (`.jpg`) with their annotations (`.json`). Each annotation contains the bounding boxes of the objects in the image; this project only uses the ones labeled `information--pedestrians-crossing--g1`.
- `templates/triangle_template.png` — reference triangle image used by `cv2.matchShapes` to recognize the pictogram shape.
