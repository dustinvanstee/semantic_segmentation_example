# semantic_segmentation_example

This is a sample implementation of semantic segmentation using kitti [http://www.cvlibs.net/datasets/kitti/eval_road.php] road dataset.  There are 2 implementations of code.   

* FCN.py - original segmentation code written in a non GPU optimized manner
* FCN_opt.py -- optimized code, removing IO from inner training loop.  

This code is useful to highlight the perils of downloading opensource implementations 'as-is'.  Typically to get the best performance you need to be careful about the data preprocessing pipeline.  See PDF for some highlights of the modifications!

---
To Run :
download kitti base data set --> http://www.cvlibs.net/download.php?file=data_road.zip and unzip to directory of your choice.

Recommend using anaconda to install your python packages and then just modify the code to point to the proper data directory ...
eg. 
```data_dir = './Data/Road/'```

Then just run 
```python FCN.py``` or ```python FCN_opt.py```

--- 
Note, if you are using PowerAI run this prior to python job.
```source /opt/DL/tensorflow/bin/tensorflow-activate```

Verify your runs are using GPU resources with ```nvidia-smi -l```
