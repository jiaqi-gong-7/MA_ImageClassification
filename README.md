# MA_ImageClassification

This repository contains the full implementation of a defect classification system for medical device components using deep convolutional neural networks. The project supports both training and real-time prediction, including a human-in-the-loop correction mechanism.

---

## 🔧 Project Structure and Workflow

### 1️⃣ Dataset Splitting
**File:** `split_dataset.py`  
- **Purpose:** Splits the collected image dataset into training, validation, and test sets.  
- **Method:** Performs random stratified splitting with a default ratio of 70% training, 15% validation, and 15% testing.  
- **When to Run:** After collecting all images that need to be classified.

---

### 2️⃣ Main Pipeline Execution
**File:** `main.py`  
- **Purpose:** This is the main entry point of the pipeline. It calls and manages the following steps:
  - Data preprocessing (`data_preparation.py`)
  - Model construction and training (`model_training.py`)
  - Model evaluation (`evaluation.py`)
  - Automated PDF report generation (`report_generator.py`)  
- **Supported Architectures:** EfficientNetB3, ResNet50, DenseNet121  
- **Output:** Trained `.h5` model files, evaluation reports, and visualizations.

---

### 3️⃣ Single Image Prediction with Human-in-the-Loop
**File:** `predict_label.py`  
- **Purpose:** Loads the best-performing model and makes predictions on a single input image.  
- **Features:**
  - Preprocessing and visualization of the prediction
  - Output formatted in both PNG (for visual inspection) and JSON (for system integration)
  - **Human-in-the-loop** mechanism: users are asked to confirm prediction correctness, and misclassified images can be archived for future retraining.

---

## 🧩 Optional Modules (Not Activated in Current Version)

- **`GUI.py`**: A prototype graphical interface for future upgrade.
- **`upload_report_to_jira.py`**: A utility script for integrating reports with JIRA, planned for future use.

---

## ⚠️ Important Notes

- The dataset and trained models are not included in this repository.
- Please **download the data and saved models separately from the shared Google Drive folder** linked: https://drive.google.com/drive/folders/1dqNFCJXvo7LHAL_veKCuveQzNF2tH9aJ?usp=sharing (to be added manually).
- Model files (e.g., `.h5`) and image samples (`.jpg`, `.png`) are excluded via `.gitignore`.

---

## 📁 Repository Summary

| File                          | Function                                                                 |
|-------------------------------|--------------------------------------------------------------------------|
| `split_dataset.py`            | Split raw images into train/val/test sets                               |
| `main.py`                     | Master script to run the full pipeline                                  |
| `data_preparation.py`         | Preprocessing and augmentation logic                                    |
| `model_training.py`           | Build and train CNNs with transfer learning                             |
| `evaluation.py`               | Evaluate model performance, output metrics and confusion matrix         |
| `report_generator.py`         | Generate classification PDF reports                                     |
| `predict_label.py`            | Predict the class of a single image, with optional feedback correction  |
| `GUI.py`                      | Not used currently, GUI prototype                                       |
| `upload_report_to_jira.py`    | Not used currently, JIRA integration script                             |

---

## 🧠 Future Improvements

- GUI support for user-friendly prediction
- Direct JIRA upload for quality documentation
- YOLO-based object detection or segmentation modules for fine-grained analysis
- Continuous learning automation using confirmed human corrections

---

## 📬 Contact

Author: Jiaqi Gong  
Institution: TUM – Chair of Data Processing  
License: MIT (or internal use only)

