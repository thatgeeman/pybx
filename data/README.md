## Visualisation methods
Loading a sample image with randomly generated logits and generated anchors.
```python3
from pybx import vis
im_array, annots, logits, color = vis._get_example((100,100), (3,3), pth='.')
```
Displaying image with annotations, anchor boxes over logits
```python3
vis.draw(im_array, anchors.tolist() + annots, color=color, logits=logits)
```
`color` can take a Dict of colors to highlight specific labels in the image. 
In this case, by default unspecified labels will be shown in white.
```python3
c = {'clock': 'green', 'frame': 'blue', 'a4': 'red'} # a4: anchor box 4
vis.draw(im_array, anchors.tolist() + annots, color=c)
```

## Notes on test image
Image obtained from USC-SIPI Image Database using:
```bash
! wget -q -O 'image.jpg' 'https://sipi.usc.edu/database/download.php?vol=misc&img=5.1.12'
```
### USC-SIPI Image Database
The USC-SIPI image database is a collection of digitized images. 
It is maintained primarily to support research in image processing, 
image analysis, and machine vision. The first edition of the 
USC-SIPI image database was distributed in 1977 and many new 
images have been added since then.

For free to use images, and further copyright information about 
the image used in this project, please check:
- [The copyright information](https://sipi.usc.edu/database/copyright.php)
- [The full image database](https://sipi.usc.edu/database/database.php)