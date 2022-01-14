# PyBx 
version

A simple python module to generate anchor
(aka default/prior) boxes for object detection
tasks. 
Calculated boxes are returned as ndarrays in `pascal_voc` format by default.

### Installataion
```bash
pip install pybx
```

### Usage
To calculate the anchor box for a single feature size and aspect ratio: 
```python
from pybx import anchor
image_sz = (300, 300, 3)
feature_sz = (10, 10)
asp_ratio = 1/2.
anchor.bx(image_sz, feature_sz, asp_ratio)
```
To calculate anchor boxes for multiple feature sizes and aspect ratios:
```python3
feature_szs = [(10, 10), (8, 8)]
asp_ratios = [1., 1/2., 2.]
anchor.bxs(image_sz, feature_szs, asp_ratios)
```
Notes on [visualising generated boxes.](data/README.md)
### Todo
- [ ] Companion notebook
- [ ] Unit tests
- [ ] Specific tests
  - `image_sz` and `feature_sz` of different aspect ratios