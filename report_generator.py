import numpy as np

def generate_report(y_true, y_pred_classes, output_file='classification_report.txt'):
    try:
        total = len(y_true)
        correct = np.sum(y_true == y_pred_classes)
        accuracy = correct / total * 100
        defective_indices = np.where(y_true != y_pred_classes)[0]

        report_content = f"""
        Total Number of Samples: {total}
        Number of Correctly Classified Samples: {correct}
        Number of Misclassified Samples: {len(defective_indices)}
        Accuracy: {accuracy:.2f}%
        Indices of Misclassified Samples: {defective_indices.tolist()}
        """
        with open(output_file, 'w') as f:
            f.write(report_content)
        print(f"Report saved to {output_file}")
    except IOError as e:
        print(f"Error saving report: {e}")

