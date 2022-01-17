# PyBx 

*WIP*

A simple python package to generate anchor
(aka default/prior) boxes for object detection
tasks. Calculated anchor boxes are returned as ndarrays in `pascal_voc` format by default.

### Installation
```shell
pip install pybx
```

### Usage
<a href="https://colab.research.google.com/github/thatgeeman/pybx/blob/master/nbs/pybx_walkthrough.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

To calculate the anchor boxes for a single feature size and aspect ratio, given the image size: 
```python
from pybx import anchor

image_sz = (300, 300, 3)
feature_sz = (10, 10)
asp_ratio = 1/2.

anchor.bx(image_sz, feature_sz, asp_ratio)
```
More on visualising the anchor boxes with `pybx` [here](data/README.md).
```python
anchor.bx(image_sz, feature_sz, asp_ratio)
```
![](data/box-1.png)

To calculate anchor boxes for multiple feature sizes and aspect ratios: 

```python
feature_szs = [(10, 10), (8, 8)]
asp_ratios = [1., 1/2., 2.]

anchor.bxs(image_sz, feature_szs, asp_ratios)
```

### Todo
- [x] Wrapper class for boxes with `vis.show()` method
- [x] Companion notebook
  - [ ] Update with new Class methods
- [x] Integrate MultiBx into anchor.bx()
- [ ] IOU check (return best overlap boxes)
- [ ] Return masks 
- [ ] Unit tests
- [x] Specific tests
  - [x] `feature_sz` of different aspect ratios
  - [x] `image_sz` of different aspect ratios
- [ ] Move to setup.py
- [ ] Generate docs


