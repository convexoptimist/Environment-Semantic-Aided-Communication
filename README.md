# Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction

This is a python code package related to the following article: S.Imran et al., "Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction", arXiv:2302.06736

# Abstract of the Article
Millimeter-wave (mmWave) and terahertz (THz) communication systems adopt large antenna arrays to ensure adequate receive signal power. However, adjusting the narrow beams of these antenna arrays typically incurs high beam training overhead that scales with the number of antennas. Recently proposed vision-aided beam prediction solutions, which utilize raw RGB images captured at the basestation to predict the optimal beams, have shown initial promising results. However, they still have a considerable computational complexity, limiting their adoption in the real world. To address these challenges, this paper focuses on developing and comparing various approaches that extract lightweight semantic information from the visual data. The results show that the proposed solutions can significantly decrease the computational requirements while achieving similar beam prediction accuracy compared to the previously proposed vision-aided solutions.

# Code Package Content

The scripts for generating the results of the ML solutions in the paper. This script adopts Scenario 5 and Scenario 7 of DeepSense6G dataset.
To reproduce the results, please follow these steps:

**Downloading Dataset and Code** 
1. Download Scenario 5 and Scenario 7 of Deepsense6G dataset.
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory

**Generating Development Dataset** 
1. To generate the bounding boxes and masks from MobileNet, run 'MobilNet_mask_generation.ipynb' and 'MobileNet_bbox_generation.ipynb'. 
2. To generate the bounding boxes and masks from Yolov7, run'Yolov7_bbox_and_masks_generating_code.ipynb'.

**ML Models Training**
1.  Run main_beam.py/main_pos_beam.py files for Scenario 5/Scenario 7 in the semantic_mask_bbox_code folder.
2.  Run 'Yolov7_Masks_to_beam_prediction.ipynb' in the YOLOv7 masks_to_beam_pred folder.

**ML Models Testing**
1.   Run the eval_main_beam.py/eval_main_pos_beam.py codes in the semantic_mask_bbox_code folder with the checkpoint obtained from the training code.
2.   Run 'Eval_Yolov7_Masks_to_beam_prediction.ipynb' in the YOLOv7 masks_to_beam_pred folder with the checkpoint obtained from the training code.

**Reproducing Results**
1.   Run the eval_main_beam.py/eval_main_pos_beam.py codes with the checkpoints provided in the semantic_mask_bbox_code folder. 
2.   Run 'Eval_Yolov7_Masks_to_beam_prediction.ipynb' with the checkpoint provided in the YOLOv7 masks_to_beam_pred folder.


# License and Referencing
This code package is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. If you in any way use this code for research that results in publications, please cite our original article:

S. Imran, G. Charan and A. Alkhateeb. “Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction.” arXiv preprint arXiv:2302.06736    (2023). 

If you use the DeepSense 6G dataset, please also cite our dataset article:

A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, J. Morais, U. Demirhan, and N. Srinivas, “DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and     Communication Dataset,” arXiv preprint arXiv:2211.09769 (2022) [Online]. Available: https://www.DeepSense6G.net





