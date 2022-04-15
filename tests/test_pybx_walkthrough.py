import unittest
import os
from operator import __mul__
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import json
from pybx import anchor
from pybx import vis
from pybx.basics import *


def test_all_nbs():
    # !/usr/bin/env python
    # coding: utf-8

    # <a href="https://colab.research.google.com/github/thatgeeman/pybx/blob/master/nbs/pybx_walkthrough.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    # >⚠ Note: walkthrough for v0.2.1 ⚠
    # >
    # >run `! pip freeze | grep pybx` to see the installed version.

    # # PyBx
    #
    # PyBx is a simple python package to generate anchor boxes (aka default/prior boxes) for object detection tasks.

    # # SSD for Object Detection
    #
    # This walkthrough is build around the [Single-Shot Detection (SSD)](https://arxiv.org/pdf/1512.02325.pdf) algorithm. The SSD can be imagined as an encoder-decoder model architecture, where the input image is fed into a `backbone` (encoder) to generate inital features, which then goes through a series of 2D convolution layers (decoders) to perform further feature extraction/prediction tasks at each layer. For a single image, each layer in the decoder produces a total of `N x (4 + C)` predictions. Here `C` is the number of classes (plus one for `background` class) in the detection task and 4 comes from the corners of the rectangular bounding box.
    #
    # ### Usage of the term Feature/Filter/Channel
    #
    # Channel: RGB dimensione, also called a Filter
    #
    # Feature: (W,H) of a single channel

    # ## Example case
    # For this example, we assume that our input image is a single channel image is of shape `[B, 3, 300, 300]` where `B` is the batch size. Assuming that a pretrained `VGG-16` is our model `backbone`, the output feature shape would be: `[B, 512, 37, 37]`. Meaning that, 512 channels of shape `[37, 37]` were extracted from each image in the batch. In the subsequent decoder layers, for simplicity we double the channels while halving the feature shape using `3x3` `stride=2` convolutions (except for first decoder layer where convolution is not applied). This results in the following shapes:
    #
    # ```python
    # torch.Size([-1, 512, 37, 37])  # inp from vgg-16 encoder
    # torch.Size([-1, 1024, 18, 18]) # first layer logits
    # torch.Size([-1, 2048, 8, 8])   # second layer logits
    # torch.Size([-1, 4096, 3, 3])   # third layer logits
    # ```
    #
    # <img src="https://lilianweng.github.io/lil-log/assets/images/SSD-box-scales.png" width="500" />

    # ## Sample image
    # Image obtained from USC-SIPI Image Database.
    # The USC-SIPI image database is a collection of digitized images. It is maintained primarily to support research in image processing, image analysis, and machine vision. The first edition of the USC-SIPI image database was distributed in 1977 and many new images have been added since then.

    # Set working directory.

    # In[ ]:

    os.environ["DATADIR"] = "./data"

    # Install package if not already present.

    # In[ ]:

    # ## About anchor Boxes
    #
    # We are expected to provide our models with "good" anchor (aka default/prior) boxes. Strong opinion: Our model is [only as good as the initial anchor boxes](https://towardsdatascience.com/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9) that we generate. Inorder to improve the coverage of our model, we tend to add additional anchor boxes of different aspect ratios. Now, for a single image, each layer in the decoder produces a total of `N x A x (4 + C)` predictions. Here `A` is the number of aspect ratios to generate additional anchor boxes.
    #
    # ### Task description
    #
    # Our aim is to find the maximum number of anchor boxes in varying sizes `feature_szs` and aspect ratios `asp_ratios` across the entire image. We apply no filtering to get rid of low (IOU) anchors.
    #
    # <img src="https://lilianweng.github.io/lil-log/assets/images/SSD-framework.png" width="600" />

    # In[ ]:

    feature_szs = [(37, 37), (18, 18), (8, 8), (3, 3)]

    # In[ ]:

    asp_ratios = [1 / 2., 1., 2.]

    # In[ ]:


    n_boxes = sum([__mul__(*f) for f in feature_szs])
    print(f'minimum anchor boxes with 1 aspect ratio: {n_boxes}')
    print(f'minimum anchor boxes with {len(asp_ratios)} aspect ratios: {n_boxes * len(asp_ratios)}')

    # # Loading an image

    # In[ ]:



    # In[ ]:

    datadir = Path(os.environ["DATADIR"])
    datadir

    # In[ ]:

    im = cv2.cvtColor(cv2.imread((datadir / "image.jpg").as_posix()), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (300, 300), interpolation=cv2.INTER_NEAREST)
    _ = plt.imshow(im)

    # In[ ]:

    im.size

    # We also make 2 truth bounding boxes `bbox` for this image around the clock and the photoframe in `pascal voc` format:

    # In[ ]:

    bbox = [dict(x_min=150, y_min=70, x_max=270, y_max=220, label='clock'),
            dict(x_min=10, y_min=180, x_max=115, y_max=260, label='frame'), ]
    bbox

    # Save annotations as a json file.

    # In[ ]:

    with open(datadir / 'annots.json', 'w') as f:
        f.write(json.dumps(bbox))

    # In[ ]:

    type(bbox[0])

    # # Using PyBx

    # In[ ]:



    image_sz = (300, 300, 3)  # W, H, C
    feature_sz = (3, 3)  # number of features along W, H
    asp_ratio = 1.  # aspect ratio of the anchor box

    anchors, labels = anchor.bx(image_sz, feature_sz, asp_ratio)

    # There are several ways to visualize the anchors. First we import the `vis` method.

    # In[ ]:



    # In[ ]:

    bbox

    # In[ ]:

    image_sz

    # ### Visualizing the locally stored `image.png` with provided bounding boxes.

    # In[ ]:

    im.size

    # In[ ]:

    plt.imshow(im)

    # In[ ]:

    image_arr = np.array(im)

    # In[ ]:

    v = vis.VisBx(image_arr=image_arr,
                  annots=bbox,
                  color={'frame': 'red', 'clock': 'blue'})

    # In[ ]:

    v.show()  # without any arguments

    # Pass arguments to `show` method to overlay with calculated anchor boxes.

    # In[ ]:

    v.show(anchors, labels)

    # ### Using the `sample=True` parameter to load a file
    # By default it looks in the current path `pth="."` for an image file `img_fn="image.png"` and annotations file `ann_fn="annots.json"`.

    # In[ ]:

    datadir

    # In[ ]:

    v = vis.VisBx(image_sz=(300, 300, 3),
                  color={'frame': 'red', 'clock': 'blue'},
                  img_fn=datadir / 'image.jpg',
                  ann_fn=datadir / 'annots.json')

    # In[ ]:

    v.show(anchors, labels)

    # ### Using randomly generated noise as `image_arr`

    # In[ ]:

    v = vis.VisBx(image_sz=(300, 300, 3))

    # In[ ]:

    v.show(anchors, labels)

    # The boxes in white are the anchor boxes. We can hightlight them with a different color by looking up specific box labels.

    # In[ ]:

    anchors.shape, labels

    # We see there are 9 labels and box coordinates reported by `anchor.bx()` for our `feature_sz=3x3` and single `asp_ratio`. Once instantiated as a `MultiBx`, we can use the `mbx()` method.

    # In[ ]:



    b = mbx(anchors, labels)  # instantiates MultiBx for us

    # In[ ]:

    type(b)

    # We can iterate over a `MultiBx` object using list comprehension to understand the internal checks:

    # In[ ]:

    [(i, b_.valid()) for i, b_ in enumerate(b)]  # only valid boxes shown

    # `b_.valid()` returned `True` meaning that the box is considered valid.
    #
    # We can also calculate the areas of these boxes.

    # Each box `b_` of the `MultiBx` b is of type `BaseBx` which has some additional methods.

    # In[ ]:

    [b_.area() for b_ in b]

    # Each `BaseBx` is also pseudo-iterable (calling an iterator returns `self` itself and not the coordinates or labels).

    # In[ ]:

    b_ = b[0]
    [x for x in b_]

    # We can also stack the `BxTypes`. Issues a `UserWarning` if we try to add `BaseBx`+`MultiBx` or `BaseBx`+`BaseBx`. This is to preserve the philosophy of a `BaseBx`, since adding something to a `BaseBx`, which should technically only hold a single coordinate and label, makes the result a `MultiBx`.

    # In[ ]:

    b_s = b_ + b_
    b_s.coords, b_s.label

    # To safely add two boxes, use `basics.stack_bxs()` method.

    # In[ ]:

    stack_bxs(b_, b_).coords

    # From `v1.0.0` `BaseBx` can be iterated. What does it mean to iterate a single coordinate. Technically it should return each point of the coordinate. But `BaseBx` behaves differently on being iterated. It returns the `BaseBx` itself.

    # In[ ]:

    [x for x in b_]

    # To truly iterate over the coordinates and label, one must do:

    # In[ ]:

    [x for x in b_.values()]

    # In[ ]:

    # or
    # [x.label for x in b_]
    [x.coords for x in b_]

    # Coming back to the `MultiBx` types, we can display the coordinates of the valid boxes:

    # In[ ]:

    [b_.coords for b_ in b]  # selected boxes only!

    # Displaying the labels of valid boxes

    # In[ ]:

    [b_.label for b_ in b]  # selected boxes only!

    # We can ofcourse see all the 16 boxes calculated by `anchor.bx()` from the `MultiBx` as well:

    # In[ ]:

    b.coords, b.label

    # > The `vis.VisBx` internally converts all coordinates in `list`/`json`/`ndarray` to a `MultiBx` and shows only `valid` boxes.
    #
    # We can also overlay the features generated by the model on the original image. `logits=True` generates random logits (`np.random.randn`) of the same shape as feature sizes for illustration purposes.
    #
    # To aid the explainability of the model, actual model logits can also be passed into the same parameter `logits` as an array or detached tensor.

    # In[ ]:

    # ask VisBx to use random logits with logits=True
    vis.VisBx(image_sz=image_sz, logits=True, feature_sz=feature_sz).show(anchors, labels)

    # In[ ]:

    # ask VisBx to use passed logits with logits=logits
    logits = np.random.randn(3, 3)  # assuming these are model logits
    logits

    # In[ ]:

    v = vis.VisBx(image_sz=image_sz, logits=logits).show(anchors, labels)

    # We can hightlight them with a different color if needed. Anchor boxes generated with `named=True` parameter automatically sets the label for each box in the format: `{anchor_sfx}_{feature_sz}_{asp_ratio}_{box_number}`. `anchor_sfx` is also an optional parameter that can be passed to `anchor.bx()`. Here we change the color of one anchor box and one ground truth box.

    # In[ ]:

    labels[4]

    # In[ ]:

    v = vis.VisBx(image_sz=image_sz)
    v.show(anchors, labels, color={'a_3x3_1.0_4': 'red', 'clock': 'orange'})

    # The box `a_3x3_1.0_4` is not fully highlighted due to overlapping edges of other anchor boxes. A quick and dirty fix to isolate the said box:

    # In[ ]:

    v.show([a for i, a in enumerate(anchors) if labels[i] == 'a_3x3_1.0_4'],
           [l for l in labels if l == 'a_3x3_1.0_4'],
           color={'a_3x3_1.0_4': 'red', 'clock': 'orange'})

    # # Working with mulitple feature sizes and aspect ratios
    # Finally we calculate anchor boxes for multiple feature sizes and aspect ratios.

    # In[ ]:

    feature_szs = [(3, 3), (2, 2)]
    asp_ratios = [1 / 2., 2.]

    anchors, labels = anchor.bxs(image_sz, feature_szs, asp_ratios)

    # This is essentially a wrapper to do list comprehension over the passed feature sizes and aspect ratios (but additionally stacks them together into an ndarray).
    #
    # ```
    # [anchor.bx(image_sz, f, ar) for f in feature_szs for ar in asp_ratios]
    # ```

    # In[ ]:

    labels[4], labels[5]

    # In[ ]:

    v = vis.VisBx(image_sz=image_sz)
    v.show(anchors, labels, color={'a_3x3_0.5_4': 'red', 'a_2x2_0.5_0': 'red'})

    # As simple as that! Do leave a star or raise issues and suggestions on the project page if you found this useful!
    #
    # Project page: [GitHub](https://github.com/thatgeeman/pybx)
    #
    # PyPi Package: [PyBx](https://pypi.org/project/pybx/)

    # In[ ]:


if __name__ == '__main__':
    unittest.main()







