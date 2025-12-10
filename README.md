# [P2P-CV @ WACV 2026] LLM Augmented Intervenable Multimodal Adaptor for Post-operative Complication Prediction in Lung Cancer Surgery
### Code for anonymous submission in P2P-CV workshop at WACV 2026

## Paper Overview
<img width="923" height="660" alt="intro_diag" src="https://github.com/user-attachments/assets/6e30f179-2c5a-4412-83d8-7acbbdcd962c" />

In this work, we present **MIRACLE (Multi-modal Integrated Radiomics And Clinical Language-based Explanation)**, a unified deep learning framework that integrates: 
- Structured clinical features collected during preoperative evaluations
- Chest CT derived radiological features
- LLM generated, evidence anchored textual explanations

Traditional methods use only clinical data and offer no explainability, making them black-box systems. The proposed model integrates clinical and radiological data with explanatory remarks, enabling transparency and intervention as a glass-box system.

## Proposed Architecture
<img width="1207" height="612" alt="architecture" src="https://github.com/user-attachments/assets/f89f141a-9c82-448e-a450-2b7adb9b4538" />

The proposed architecture consists of three main modules:
- Two separate Bayesian MLP networks, one each for clinical and radiological features
- An encoding module using a frozen encoder, fine-tuned on medical data for textual remarks
- A fusion network for final prediction

The processing pipeline is illustrated in the paper.

## Installations and environment creation
```
conda create -n miracle python=3.9.21
conda activate miracle
pip install -r requirements.txt
```
We used PyTorch==2.2.2 for CUDA=12.2

## Preparing data and pretrained checkpoints

### Datasets used in training, validation and testing
The Proposed dataset, called POC-L is acquired from real lung cancer surgery patients, which went through a surgery at a well reknowned cancer research hospital.

**Dataset statistics:**
- 3094 patients that have went through Lung cancer surgery from 2009-2023, in well reknowned hospital and cancer research institute
- Patients were split into training (2,694; 22.6% complications), validation (200; 47.5%) and testing (200; 53.5%) splits
- 57% of patients are female and 43% male
- Dataset is dominated with White ethnicity patients with representations from African-American and Asian populations
- All records were de-identified prior to analysis and the study was approved under IRB protocol BDR 176423
- Each case has 17 structured preoperative clinical variables and 113 standardized radiomic features
- Postoperative complications were defined across ten major complication events curated by domain experts
- the presence of any of the ten complication events was aggregated to produce a binary global outcome label indicating whether a patient experienced at least one postoperative complication

### Preprocessing the data
- Preprocessing is done using "preprocess_input_data.py" script, located in the dataset folder
- Continuous clinical features were normalized using Min-Max scaling fitted on the training split, while categorical variables were label encoded
- Radiomic features were standardized to ensure numerical stability

### Pretrained models and Finetuned checkpoints
Download the zip file from [Link](https://drive.google.com/file/d/1BUGv8cFfFLRkmhflzNGgPDKml0qFLgO5/view?usp=sharing) and unzip the contents in main directory to use it in evaluation and training scripts

## Training
- MIRACLE is trained using the train.py script on the training split of POC-L and validated on the validation split

## Testing and Evaluation
- The trained model checkpoint is evaluated on the testing split of POC-L using inference.py script

## Qualitative Analysis of the LLM generated remarks
The remarks from LLM is compared against remarks given by Surgeons on testing split of POC-L. We employed three distinct types of LLMs (instruction-based, reasoning, and fine-tuned) for remark generation. Despite being given the same set, the remarks varied across the models. To quantify the relative quality of LLM explanations, we carried out a two-stage evaluation:

1. **Automated Adjudication:** The comparison is done using LLM as a judge. It can be seen that most of the remarks generated from LLM are completely aligned to remarks given by LLM.
<img width="800" height="600" alt="remark_alignment" src="https://github.com/user-attachments/assets/7d45b519-e1e1-4eb2-b59c-8dc0e99419af" />

2. **Expert Manual Review:** A panel of thoracic surgery specialists inspected paired surgeon and LLM-generated remarks for a representative subset of test cases. They labeled each LLM explanation as:
- Performs better
- Performs comparably
- Performs worse

<img width="2168" height="574" alt="examples_final" src="https://github.com/user-attachments/assets/0502c8f8-5919-4478-8b27-8da2fdc0a408" />

## Quantitative Results
### Performance across different models
|Model | AUC(%) | TAR(%)@FAR=0.2 | TAR(%)@FAR=0.3 |
| :---: | :---: | :---: | :---: |
|Llama 3.3 70B-Instruct | 69.68  | 41.12 |  74.77 |
|DeepSeek R1-Distill Qwen-32B  |  64.49 | 54.21  |  56.07 |
|OpenBioLLM-70B  | 71.01  | 52.34  |  60.75 |
|Multivariate logistic regression   | 80.89   | **73.83** | 80.37   |
|Random Forest Classifier   | 77.00 | 62.62   | 74.76  |  
|XGBoost  | 75.17  | 53.27  | 64.48  |
|Gradient Boosting Classifier  | 78.53  | 65.42 | 67.29  |
|LightGBM  | 74.77  | 46.73 | 69.16  |
|Surgeons  | &hyphen;  | 44.86  | &hyphen;  |
|**MIRACLE (DeepSeek R1 distill)**  | 80.94  | **73.83**  | **81.31**  |
|**MIRACLE (Llama 3.3 70B-Instruct)**  | 80.84  | 71.03  | **81.31**  |
|**MIRACLE (OpenBioLLM-70B)**  | **81.04**  | 71.96  | **81.31** |

### ROC for all the models
<img width="1000" height="1000" alt="combined_ROC" src="https://github.com/user-attachments/assets/1112cb8d-6a41-4f85-8320-7aa72283b7fc" />

### Confusion Matrix for surgeons' performance
<img width="700" height="500" alt="Confusion_matrix" src="https://github.com/user-attachments/assets/6bd4e56d-b22f-4f46-9e99-00412e38f4f3" />

## Ablation Study (to analyze the contribution of each module)
|Clinical | Radiological | LLM Remarks module | AUC(%) | TAR(%)@FAR=0.2 | TAR(%)@FAR=0.3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **&check;** | **&hyphen;** | **&hyphen;** | 74.81 | 57.94 | 66.35 |
| **&check;** | **&check;** | **&hyphen;** | 78.64 | 64.48 | 76.64 |
| **&check;** | **&check;** | **&check;** | **80.94** | **73.83** | **81.31** |

## Contact
For more information or any questions, feel free to reach us at anonymouspeerblind@gmail.com

## License
MIRACLE is CC-BY-NC 4.0 licensed, as found in the LICENSE file. It is released for academic research / non-commercial use only.
