# PyBx
[![PyPI version](https://badge.fury.io/py/pybx.svg)](https://badge.fury.io/py/pybx)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thatgeeman/pybx/blob/master/nbs/pybx_walkthrough.ipynb)

A simple python package to generate anchor
(aka default/prior) boxes for object detection
tasks. Calculated anchor boxes are in `pascal_voc` format by default.

### Installation
```shell
pip install pybx
```

### Usage

To calculate the anchor boxes for a single feature size and 
aspect ratio, given the image size: 
```python
from pybx import anchor

image_sz = (300, 300, 3)
feature_sz = (10, 10)
asp_ratio = 1/2.

coords, labels = anchor.bx(image_sz, feature_sz, asp_ratio)
```
![](data/box-1.png)

### Introducing `MultiBx`
To calculate anchor boxes for multiple feature sizes and 
aspect ratios: 

```python
feature_szs = [(10, 10), (8, 8)]
asp_ratios = [1., 1/2., 2.]

coords, labels = anchor.bxs(image_sz, feature_szs, asp_ratios)
```
All anchor boxes are returned as ndarrays of shape `(N,4)` where N
is the number of boxes along with [default labels](data/README.md). These and other types (`list, json`) of box annotations
can be instantialized as a `MultiBx`, exposing many useful methods and attributes
of each anchor box. For example to calculate the area of each box
iteratively:
```python
from pybx.basics import * 
boxes = mbx(coords, labels) 
areas = [b.area() for b in boxes]
```
Objects of the type `MultiBx` can also be "added" which stacks 
them into a new `MultiBx`:
```python
boxes_true = mbx(coords_json)    # annotation as json records
boxes_anchor = mbx(coords_numpy) # annotation as ndarray
boxes = boxes_true+boxes_anchor
```

The `vis` module of `pybx` can be used to visualize these "stacks"
of `MultiBx`, raw `ndarray`/`list`/`json` records, 
target annotations and 
model logits. Please rerer 
to [Visualising anchor boxes](data/README.md).

## Todo
- [x] Wrapper class for boxes with `VisBx.show()` method
- [x] Companion notebook
  - [x] Update with new Class methods
- [x] Integrate MultiBx into anchor.bx()
- [x] IOU calcultaion
- [x] Unit tests
- [x] Specific tests
  - [x] `feature_sz` of different aspect ratios
  - [x] `image_sz` of different aspect ratios
- [ ] Generate docs `sphinx`
- [ ] clean docstrings


