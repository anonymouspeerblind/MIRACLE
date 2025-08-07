# [KDD'26] LLM Augmented Intervenable Multimodal Adaptor for Post-operative Complication Prediction in Lung Cancer Surgery
### Code for anonymous submission in 32nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2026

## Project Overview
<img width="923" height="660" alt="kdd_intro" src="https://github.com/user-attachments/assets/6e289d26-9a22-406a-a420-4799694e0beb" />

In this work, we present MIRACLE (Multi-modal Integrated Radiomics And Clinical Language-based Explanation), a unified deep learning framework that integrates: (i) Structured clinical features collected during preoperative evaluations, (ii) Chest CT–derived radiological features, and (iii) LLM-generated, evidence-anchored textual explanations.  To our knowledge, MIRACLE is the first approach to jointly fuse these modalities to predict postoperative complications in lung cancer surgery while offering both interpretability and clinician-intervenability.
Traditional methods use only clinical data and offer no explainability, making them black-box systems. The proposed model integrates clinical and radiological data with explanatory remarks, enabling transparency and intervention as a glass-box system.

<img width="1207" height="612" alt="kdd_architecture" src="https://github.com/user-attachments/assets/e73d9423-8853-4ef7-8091-4aabb95709df" />

## Installations and environment creation
- conda create -n miracle python=3.9.21
- conda activate miracle
- pip install -r requirements.txt

We used PyTorch==2.2.2 for CUDA=12.2

## Preparing data and pretrained checkpoints

### Datasets used in training, validation and testing

<img width="1653" height="871" alt="dataset_example_kdd" src="https://github.com/user-attachments/assets/8db573f5-0816-4dcb-97d5-487a54a24597" />

- 3094 patients that have went through Lung cancer surgery from 2009-2023, in well reknowned hospital and cancer research institute
- Training split has 2694 patients
- Validation and Testing split has 200 patients each

<img width="612" height="180" alt="overall_distribution" src="https://github.com/user-attachments/assets/fd4f3786-aef2-44d3-b033-e8a51e282d69" />

### Preprocessing the data
- Preprocessing is done using "preprocess_input_data.py" script, located in the dataset folder
- Data is standerdized and normalized

### Pretrained models and Finetuned checkpoints
Download the zip file from [Link](https://drive.google.com/file/d/1BUGv8cFfFLRkmhflzNGgPDKml0qFLgO5/view?usp=sharing) and unzip the contents in main directory to use it in evaluation and training scripts

## Training
- MIRACLE is trained using the train.py script on the training dataset and validated on the validation dataset

## Testing and Evaluation
- The trained model checkpoint is evaluated on the testing dataset using inference.py script

## Surgeons vs LLM
- The remarks from LLM is compared against remarks given by Surgeons on testing dataset
- The comparison is done using LLM as a judge
- It can be seen that most of the remarks generated from LLM are completely aligned to remarks given by LLM
- Further strengthens our model

<img width="900" height="900" alt="Combined_human_vs_llm" src="https://github.com/user-attachments/assets/40a3cee8-3946-4a8c-a8da-09fa9af0f5db" />
<img width="1337" height="530" alt="examples_final" src="https://github.com/user-attachments/assets/67774632-a874-4fc9-95ba-b227f0fd598d" />


## Performance across different models
|Model | AUC(%) | TAR(%)@FAR=0.2 | TAR(%)@FAR=0.3 |
| :---: | :---: | :---: | :---: |
|Llama 3.3 70B-Instruct (only clinical data) | 50.89 | 36.45 | 36.45 |
|DeepSeek R1-Distill Qwen-32B (only clinical data) | 62.92 | 33.64 | 66.35 |
|OpenBioLLM-70B (only clinical data) | 62.42 | 41.12 | 44.85 |
|Multivariate Logistic Regression | 80.89 | **73.83** | 80.37 |
|Random Forest Classifier | 77.00 | 62.61 | 74.76 |
|XGBoost | 75.17 | 53.27 | 64.48 |
|Gradient Boosting Classifier | 78.53 | 65.42 | 67.29 |
|LightGBM | 74.76 | 46.73 | 69.16 |
|Surgeons | &hyphen; | 44.86 | &hyphen; |
|MIRACLE (Deep Seek R1 distill) | 80.78 | 68.22 | 79.44 |
|MIRACLE (Llama 3.3 70B-Instruct) | **80.96** | 68.22 | **80.37** |
|MIRACLE (OpenBioLLM-70B) | 80.69 | 69.16 | 78.50 |

## ROC for all the models

<img width="1000" height="1000" alt="combined_ROC" src="https://github.com/user-attachments/assets/f8c3573e-3ce2-48b2-8be3-ec1033aa9a91" />


## Ablation Study (to analyze the contribution of each module)
|Clinical | Radiological | LLM Remarks module | AUC(%) | TAR(%)@FAR=0.2 | TAR(%)@FAR=0.3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| &check; | &hyphen; | &hyphen; | 74.81 | 57.94 | 66.35 |
| &check; | &check; | &hyphen; | 79.64 | 64.48 | 76.64 |
| &check; | &check; | &check; | **80.96** | **68.22** | **80.37** |

## Contact
For more information or any questions, feel free to reach us at anonymouspeerblind@gmail.com

## License
MIRACLE is CC-BY-NC 4.0 licensed, as found in the LICENSE file. It is released for academic research / non-commercial use only.
