# Learning Enriched Feature Descriptor for Image Matching and Visual Measurement
demo and pre-trained weight of HIFT --- a dense descriptor for local feature matching.
![HIFT](/samples/model.png)
Our work is accepted by IEEE Transactions on Instrumentation & Measurement 2023
# Pre-Trained Weights
We trained our HIFT with one-stage end-to-end triplet training strategy on MS-COCO, Multi-illumination and VIDIT datasets (same as LISRD) and the pre-trained weight is available at [hift](https://drive.google.com/file/d/1uOLlh--rT6_UY5VmA7-oFypCemoQJ527/view?usp=sharing)

# Model file
The core implementation of HIFT is shown in HIFT_core.py

# DEMO：SIFT+HIFT
1. We provide the image matching demo of using SIFT keypoints and HIFT descriptor in demo.ipynb.
2. We provide the demo of exporting SIFT keypoints and HIFT descriptor in export_descriptor_sift.py, and it can be easily modified to other off-the-shelf detectors and matchers for evaluation. 
```
CUDA_VISIBLE_DEVICES=0 python export_descriptor_sift.py
```
For more evaluation details, please refer to the [LISRD](https://github.com/rpautrat/LISRD)

# Citation

If you are interested in this work, please cite the following work:

```
@ARTICLE{9761834,
  author={Rao, Yuan and Yang, Jian and Ju, Yakun and Li, Cong and Rigall, Eric and Fan, Hao and Dong, Junyu},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Learning General Feature Descriptor for Visual Measurement With Hierarchical View Consistency}, 
  year={2022},
  volume={71},
  number={},
  pages={1-12},
  doi={10.1109/TIM.2022.3169563}}
```

# Acknowledgments
Our work is based on [LISRD](https://github.com/rpautrat/LISRD) and we use their code.  We appreciate the previous open-source repository [LISRD](https://github.com/rpautrat/LISRD)
