# report_generator.py 
from fpdf import FPDF
import os

def generate_report(test_acc, classification_rep, conclusion_text, cm_path, roc_path, class_names=None, save_path="results/reports/report_summary.pdf"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Product Classification Model Evaluation Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Model accuracy on test set: {test_acc*100:.2f}%\n")

    # 红绿灯机制说明图例（含彩色圆点与文字说明）
    if class_names:
        color_map = {
            "complete product": ((0, 200, 0), "GREEN (Qualified)"),
            "valid pcb": ((0, 200, 0), "GREEN (Qualified)"),
            "missing cd": ((255, 0, 0), "RED (Defective)"),
            "no ropes": ((255, 0, 0), "RED (Defective)"),
            "no strain relief": ((255, 0, 0), "RED (Defective)")
        }
        default_color = ((255, 165, 0), "YELLOW (Possibly Defective)")

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Category-Based Quality Classification (Color Legend):", ln=True)
        pdf.set_font("Arial", size=11)

        for cls in class_names:
            color_rgb, meaning = color_map.get(cls.lower(), default_color)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_fill_color(*color_rgb)
            pdf.ellipse(x + 2, y + 2, 5, 5, style='F')
            pdf.set_xy(x + 10, y)
            pdf.cell(0, 7, f"{cls}: {meaning}", ln=True)
        pdf.ln(4)


    pdf.ln(5)
    pdf.multi_cell(0, 8, "Classification Report:\n")
    pdf.set_font("Courier", size=10)
    pdf.multi_cell(0, 5, classification_rep)

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Automated Conclusion:\n{conclusion_text}")

    if os.path.exists(cm_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Confusion Matrix", ln=True)
        pdf.image(cm_path, w=180)
    else:
        print(f"Confusion matrix not found at {cm_path}")

    if os.path.exists(roc_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "ROC Curve", ln=True)
        pdf.image(roc_path, w=180)
    else:
        print(f"ROC curve not found at {roc_path}")


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pdf.output(save_path)
    print(f"PDF report generated: {save_path}")
    return save_path

def generate_conclusion(report_dict):
    if 'Valid PCB' in report_dict:
        recall_valid_pcb = report_dict['Valid PCB']['recall']
        if recall_valid_pcb < 0.85:
            return "Warning: The model struggles to identify complete products. Consider collecting more samples or retraining the model."
        else:
            return "Conclusion: The model performs well. The classification result for complete products is reliable."
    return "'Valid PCB' class not found in the report. Please check the results."

