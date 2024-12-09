# main.py

from data_preparation import create_generators
from model_training import build_model, train_model
from evaluation import evaluate_model
from report_generator import generate_report

# ------------------ Step 1: Data Preparation ------------------
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Preparing data generators...")
train_generator, val_generator, test_generator = create_generators(
    train_dir, val_dir, test_dir, IMG_SIZE, BATCH_SIZE
)

# ------------------ Step 2: Model Construction and Training ------------------
input_shape = (224, 224, 3)
num_classes = train_generator.num_classes

print("Building the model...")
model = build_model(input_shape, num_classes)

print("Training the model with fine-tuning...")
# Set fine_tune=True to enable fine-tuning
history, history_fine = train_model(model, train_generator, val_generator, fine_tune=True, fine_tune_epochs=5)

# ------------------ Step 3: Model Evaluation ------------------
print("Evaluating the model...")
y_true, y_pred_classes = evaluate_model(model, test_generator)

# ------------------ Step 4: Report Generation ------------------
output_file = 'classification_report.txt'
print(f"Generating the report at {output_file}...")
generate_report(y_true, y_pred_classes, output_file)

print("Workflow completed successfully!")

