# Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction

This is a python code package related to the following article: S.Imran et al., "Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction", arXiv:2302.06736

# Abstract of the Article
Millimeter-wave (mmWave) and terahertz (THz) communication systems adopt large antenna arrays to ensure adequate receive signal power. However, adjusting the narrow beams of these antenna arrays typically incurs high beam training overhead that scales with the number of antennas. Recently proposed vision-aided beam prediction solutions, which utilize raw RGB images captured at the basestation to predict the optimal beams, have shown initial promising results. However, they still have a considerable computational complexity, limiting their adoption in the real world. To address these challenges, this paper focuses on developing and comparing various approaches that extract lightweight semantic information from the visual data. The results show that the proposed solutions can significantly decrease the computational requirements while achieving similar beam prediction accuracy compared to the previously proposed vision-aided solutions.

# Code Package Content

The scripts for generating the results of the ML solutions in the paper. This script adopts Scenario 5 and Scenario 7 of DeepSense6G dataset.
To reproduce the results, please follow these steps:

**Download Dataset and Code** 

Download Scenario 5 and Scenario 7 of Deepsense6G dataset.
Download (or clone) the repository into a directory.
Extract the dataset into the repository directory

**Generate Development Dataset** 

Run 'MobilNet_mask_generation.ipynb', 'MobileNet_bbox_generation.ipynb' and 'Yolov7_bbox_and_masks_generating_code.ipynb' separately for scenario 5 and scenario 7 of the Deepsense 6G dataset to generate the bounding boxes and masks from MobileNet v2 and Yolov7.

**ML Models Training**

To train the ML models (except the ML model for predicting beam index using masks from YOLOv7), run main_beam.py or main_pos_beam.py files for both Scenario 5 and Scenario 7 in the semantic_mask_bbox_code folder. 
To reproduce the results, run the eval_main_beam.py or eval_main_pos_beam.py codes in the semantic_mask_bbox_code folder. 

**Training the Ml Model for predicting beam index from the masks of mobile units given by Yolov7**
For training this ML model, run separately for both scenario 5 and scenario 7 'Yolov7_Masks_to_beam_prediction.ipynb' in the YOLOv7 masks_to_beam_pred folder . To reproduce the results, run 'Eval_Yolov7_Masks_to_beam_prediction.ipynb' in the same folder.

# License and Referencing
This code package is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. If you in any way use this code for research that results in publications, please cite our original article:

S. Imran, G. Charan and A. Alkhateeb. “Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction.” arXiv preprint arXiv:2302.06736    (2023). 

If you use the DeepSense 6G dataset, please also cite our dataset article:

A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, J. Morais, U. Demirhan, and N. Srinivas, “DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and     Communication Dataset,” arXiv preprint arXiv:2211.09769 (2022) [Online]. Available: https://www.DeepSense6G.net





