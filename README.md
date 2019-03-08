# semantic_segmentation_example  [Work in Progress]

This is a sample implementation of semantic segmentation using kitti [http://www.cvlibs.net/datasets/kitti/eval_road.php] road dataset.  There are 2 implementations of code.   

* FCN.py - original segmentation code written in a non GPU optimized manner
* FCN_opt.py -- optimized code, removing IO from inner training loop.  

This code is useful to highlight the perils of downloading opensource implementations 'as-is'.  Typically to get the best performance you need to be careful about the data preprocessing pipeline.  See PDF for some highlights of the modifications!

---
To Run : \
0. Setup python environment (see a couple platform specifics below)
1. Clone git repo to directory of your choice \
  ```git clone https://github.com/dustinvanstee/semantic_segmentation_example.git```
2. ```cd ./semantic_segmentation_example/Semantic_Segmentation```
3. make Data/Road directory under Semantic_Segmentation \
```mkdir Data/Road```
4. Change directory to Data/Road \
``` cd Data/Road```
5. download kitti base data set --> http://www.cvlibs.net/download.php?file=data_road.zip and unzip to directory of your choice.  Preferably, this would be in a ```Data/Road``` directory under the git repo 

6. get vgg model\
```python get_vgg.py``` # writes output to Data/vgg

7. Run workload \
```python FCN.py``` or ```python FCN_opt.py```

--- 
Note, if you are using PowerAI run this prior to python job.
```source /opt/DL/tensorflow/bin/tensorflow-activate```

Verify your runs are using GPU resources with ```nvidia-smi -l```

--- 
Note, if you are using AWS run this prior to python job
[Tested with Deep Learning AMI (Ubuntu) Version 21.2 with p3.2xlarge]\
```source source activate tensorflow_p36```\
```pip install tqdm```\

---

