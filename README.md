#deeplab-series-tensorflow version

### Introduction

This is a project which aims to reimplement the deeplab series network for semantic segmentation.

### Preparation

for using this code, you have to do something else:

##### 1. Install pydensecrf

For using the densecrf in python, we turn to the project [pydensecrf](https://github.com/lucasb-eyer/pydensecrf). And you just using the following code to install it.

> pip install pydensecrf

##### 2. Download the data and model

1. for pascal data, please referring to its [official website](http://host.robots.ox.ac.uk/pascal/VOC/). Just download it and extract in the data/ .
2. for the init model,
    lfov_vgg16: [google_driver](https://drive.google.com/open?id=1MtbE1b6R4i28KabS-s7NcL08EpV3qOGl), [baidu_netdisk](https://pan.baidu.com/s/19kVdLCRGPIz05ETZuIfa7w)
   or you can download the caffemodel from the corresponding official website and convert it using the [tool](https://github.com/xtudbxk/convert_caffemodel_to_npy).

### Training

then, you just input the following sentence to train it.

> python deeplab_largefov.py <gpu_id>

### Result
deeplab_largefov -- using multiscale input and multiscale test, the final miou is 70.0% which is equally to the paper.
