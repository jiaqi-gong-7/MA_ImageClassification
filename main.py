# main.py
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import logging
import os
from data_preparation import create_generators
from model_training import build_model, train_model, save_model
from evaluation import evaluate_model
from report_generator import generate_report, generate_conclusion
from upload_report_to_jira import upload_to_jira
from sklearn.metrics import classification_report

# ------------------ Setup Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = os.path.dirname(os.path.abspath(__file__))
print(tf.config.list_physical_devices('GPU'))

# ------------------ Hyperparameters ------------------
train_dir = os.path.join(base_dir, 'data/train')
val_dir = os.path.join(base_dir, 'data/val')
test_dir = os.path.join(base_dir, 'data/test')
report_dir = os.path.join(base_dir, 'results/reports')
roc_path = os.path.join(report_dir, "roc_curve.png")
cm_path = os.path.join(report_dir, "confusion_matrix.png")
report_path = os.path.join(report_dir, 'report_summary.pdf')

IMG_SIZE = (300, 300)
BATCH_SIZE = 64
EPOCH = 30
FINE_TUNE_EPOCH = 15
input_shape = (300, 300, 3)

# ------------------ Step 1: Data Preparation ------------------
logging.info("Preparing data generators with data augmentation...")
train_generator, val_generator, test_generator, class_names, num_classes, num_train_samples, num_val_samples = create_generators(
    train_dir, val_dir, test_dir, IMG_SIZE, BATCH_SIZE)

# ------------------ Step 2: Model Construction and Training ------------------
logging.info("Building the model...")
model = build_model(input_shape, num_classes)
model.summary()  

logging.info("Training the model with fine-tuning...")
history, history_fine = train_model(
    model, train_generator=train_generator, val_generator=val_generator, num_train_samples=num_train_samples, 
    num_val_samples=num_val_samples, batch_size=BATCH_SIZE, epochs=EPOCH, fine_tune=True, fine_tune_epochs=FINE_TUNE_EPOCH)

print("Save the model...")
save_model(model)

# ------------------ Step 3: Model Evaluation ------------------
logging.info("Evaluating the model...")
y_true, y_pred_classes = evaluate_model(model, test_generator, class_names)

# ------------------ Step 4: Report Generation ------------------
logging.info("Generating PDF report...")
report_dict = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True, zero_division=1)
report_text = classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=1)
conclusion = generate_conclusion(report_dict)

test_acc = report_dict['accuracy']

generate_report(
    test_acc=test_acc,
    classification_rep=report_text,
    conclusion_text=conclusion,
    cm_path=cm_path,
    roc_path=roc_path,
    class_names=class_names,
    save_path=report_path
)

# ------------------ Step 5: Upload to JIRA  ------------------
# jira_url = "https://your-jira-instance.atlassian.net"
# jira_user = "your-email@example.com"
# jira_token = "your-jira-api-token"
# issue_key = "PROJ-123"  # Replace with your JIRA issue key

# try:
#     logging.info("Uploading PDF report to JIRA...")
#     upload_status = upload_to_jira(issue_key, [report_path], jira_url, jira_user, jira_token)
#     for file, status in upload_status.items():
#         logging.info(f"{file}: {status}")
#     logging.info("JIRA upload completed.")
# except Exception as e:
#     logging.error(f"Failed to upload report to JIRA: {e}")
