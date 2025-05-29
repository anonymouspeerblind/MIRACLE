# LLM Augmented Intervenable Multimodal Adaptor for Post-operative Complication Prediction in Lung Cancer Surgery
### Code for anonymous submission in 8th AAAI/ACM Conference on AI, Ethics, and Society 2025

![miccai_architecture](https://github.com/user-attachments/assets/01217517-11d7-4f23-966f-523c6105a8e3)

## Installations and environment creation
- conda create -n miracle python=3.9.21
- conda activate miracle
- pip install -r requirements.txt

We used PyTorch==2.2.2 for CUDA=12.2

## Preparing data and pretrained checkpoints

### Datasets used in training, validation and testing
- 3094 patients that have went through Lung cancer surgery from 2009-2023, in well reknowned hospital and cancer research institute
- Training split has 2694 patients
- Validation and Testing split has 200 patients each

![overall_distribution](https://github.com/user-attachments/assets/249cb466-7c4c-4c1d-93a6-abcd3ccae9eb)

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

![Combined_human_vs_llm](https://github.com/user-attachments/assets/67ac47a1-7870-466b-a6f3-2b273e216151)

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

![combined_ROC](https://github.com/user-attachments/assets/92f69e7b-a513-46cd-949e-c0c792688df5)

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
