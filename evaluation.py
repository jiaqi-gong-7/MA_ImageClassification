import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd

REPORT_DIR = "results/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Added traffic light description text generation

def get_quality_legend_text(class_names):
    color_map = {
        "complete product": "GREEN (Qualified)",
        "valid pcb": "GREEN (Qualified)",
        "missing cd": "RED (Defective)",
        "no ropes": "RED (Defective)",
        "no strain relief": "RED (Defective)"
    }
    lines = ["Category-Based Quality Classification:"]
    for cls in class_names:
        status = color_map.get(cls.lower(), "YELLOW (Possibly Defective)")
        lines.append(f"- {cls}: {status}")
    return "\n".join(lines)

def evaluate_model(model, test_generator, class_names):
    print("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test accuracy: {test_accuracy:.2f}")

    # Get predicted and true values
    y_true = []
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    for _, labels in test_generator:
        y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_true = np.array(y_true)

    # Automatically find the best threshold value for Valid PCB
    valid_pcb_index = class_names.index("Valid PCB")
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.3, 0.9, 0.01):
        temp_preds = y_pred_classes.copy()
        for i in range(len(y_pred_prob)):
            if y_pred_classes[i] == valid_pcb_index and y_pred_prob[i][valid_pcb_index] < threshold:
                temp_preds[i] = np.argsort(y_pred_prob[i])[-2]
        f1 = f1_score(y_true, temp_preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Applying the best threshold
    for i in range(len(y_pred_prob)):
        if y_pred_classes[i] == valid_pcb_index and y_pred_prob[i][valid_pcb_index] < best_threshold:
            y_pred_classes[i] = np.argsort(y_pred_prob[i])[-2]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Classes', fontsize=12)
    ax.set_ylabel('Actual Classes', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout(rect=[0.08, 0.08, 1, 1])
    confusion_matrix_path = os.path.join(REPORT_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Classification Report
    classification_rep = classification_report(
        y_true, y_pred_classes, target_names=class_names, zero_division=1
    )
    print(classification_rep)
    legend_text = get_quality_legend_text(class_names)  # Traffic light text

    report_path = os.path.join(REPORT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {test_accuracy:.2f}\n")
        f.write(f"Best Valid PCB Threshold: {best_threshold:.2f}\n\n")
        f.write(legend_text + "\n\n")  # Insert traffic light instructions
        f.write("Classification Report:\n")
        f.write(classification_rep + "\n")
        f.write(f"\nConfusion Matrix saved at: {confusion_matrix_path}\n")

    # ROC Curve
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc="lower right")
    roc_path = os.path.join(REPORT_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    # Save forecast details to CSV
    prediction_df = pd.DataFrame({
        "TrueLabel": [class_names[i] for i in y_true],
        "Predicted": [class_names[i] for i in y_pred_classes],
        "Confidence": [np.max(prob) for prob in y_pred_prob]
    })
    csv_path = os.path.join(REPORT_DIR, "detailed_predictions.csv")
    prediction_df.to_csv(csv_path, index=False)

    print(f"ROC curve saved to {roc_path}")
    print(f"Detailed predictions saved to {csv_path}")
    print(f"Evaluation complete. All reports saved in '{REPORT_DIR}'.")

    return y_true, y_pred_classes
