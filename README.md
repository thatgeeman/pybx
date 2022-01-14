# PyBx 

*WIP*

A simple python module to generate anchor
(aka default/prior) boxes for object detection
tasks. Calculated anchor boxes are returned as ndarrays in `pascal_voc` format by default.

### Installation
```shell
pip install pybx
```

### Usage
To calculate the anchor boxes for a single feature size and aspect ratio, given the image size: 
```python
from pybx import anchor

image_sz = (300, 300, 3)
feature_sz = (10, 10)
asp_ratio = 1/2.

anchor.bx(image_sz, feature_sz, asp_ratio)
```

To calculate anchor boxes for multiple feature sizes and aspect ratios:
```python
feature_szs = [(10, 10), (8, 8)]
asp_ratios = [1., 1/2., 2.]

anchor.bxs(image_sz, feature_szs, asp_ratios)
```
More on visualising the anchor boxes [here](data/README.md).

### Todo
- [ ] Wrapper class for boxes with `vis.draw()` method
- [ ] Companion notebook
- [ ] IOU check (return best overlap boxes)
- [ ] Return masks 
- [ ] Unit tests
- [ ] Specific tests
  - [x] `feature_sz` of different aspect ratios
  - [ ] `image_sz` of different aspect ratios
- [ ] Move to setup.py


