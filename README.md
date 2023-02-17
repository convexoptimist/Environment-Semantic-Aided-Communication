# Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction

This is a python code package related to the following article: S.Imran et al., "Environment Semantic Aided Communication: A Real World Demonstration for Beam Prediction", arXiv:2302.06736

# Abstract of the Article
Millimeter-wave (mmWave) and terahertz (THz) communication systems adopt large antenna arrays to ensure adequate receive signal power. However, adjusting the narrow beams of these antenna arrays typically incurs high beam training overhead that scales with the number of antennas. Recently proposed vision-aided beam prediction solutions, which utilize raw RGB images captured at the basestation to predict the optimal beams, have shown initial promising results. However, they still have a considerable computational complexity, limiting their adoption in the real world. To address these challenges, this paper focuses on developing and comparing various approaches that extract lightweight semantic information from the visual data. The results show that the proposed solutions can significantly decrease the computational requirements while achieving similar beam prediction accuracy compared to the previously proposed vision-aided solutions.

# Code Package Content
The scripts for generating the results of the ML solutions in the paper. This script adopts Scenario 5 and Scenario 7 of DeepSense6G dataset.
To reproduce the results, please follow these steps:

**Download Dataset and Code**
Download Scenario 5 and Scenario 7 of Deepsense6G dataset.
Download (or clone) the repository into a directory.
Extract the dataset into the repository directory

**Generate Development Dataset**
Run 'MobilNet_mask_generation.ipynb', 'MobileNet_bbox_generation.ipynb' and 'Yolov7_bbox_and_masks_generating_code.ipynb' separately for scenario 5 and scenario 7 of the Deepsese 6G dataset to generate the bounding boxes and masks from MobileNet v2 and Yolov7.

**ML Model Training**
Run main_beam.py files for both Scenario 5 and Scenario 7 in the semantic_mask_bbox_code folder. 
This code will train the ML model, save the checkpoint and analysis files

If you have any questions regarding the code and used dataset, please contact contact [Shoaib Imran](s.imran@asu.edu).





