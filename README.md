# Learning Enriched Feature Descriptor for Image Matching and Visual Measurement
demo and pre-trained weight of HIFT --- a dense descriptor for local feature matching.

# Pre-Trained Weights
We trained our AANet with one-stage end-to-end triplet training strategy on MS-COCO, Multi-illumination and VIDIT datasets (same as LISRD) and the pre-trained weight is compressed as hift.rar

# Model file
The core implementation of HIFT is shown in HIFT_core.py

# DEMO：SIFT+HIFT
1. We provide the image matching demo of using SIFT keypoints and HIFT descriptor in sift_hift.ipynb.
2. We provide the demo of exporting SIFT keypoints and AANet descriptor in export_descriptor_sift.py, and it can be easily modified to other off-the-shelf detectors and matchers for evaluation. 
```
CUDA_VISIBLE_DEVICES=0 python export_descriptor_sift.py
```
For more evaluation details, please refer to the [LISRD]: https://github.com/rpautrat/LISRD 
