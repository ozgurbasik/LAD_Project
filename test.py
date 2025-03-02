import os
import cv2
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# [Previous KITTIDataset implementation remains the same]
class KITTIDataset:
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
    def parse_kitti_label(self, label_file):
        boxes = []
        classes = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 15:
                    continue
                    
                obj_class = parts[0]
                bbox = [float(parts[4]), float(parts[5]), 
                       float(parts[6]), float(parts[7])]
                
                boxes.append(bbox)
                classes.append(obj_class)
                
        return np.array(boxes), classes
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        label_file = self.label_dir / f"{img_file.split('.')[0]}.txt"
        
        image = cv2.imread(str(self.image_dir / img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, classes = self.parse_kitti_label(label_file)
        
        return {
            'image': image,
            'boxes': boxes,
            'classes': classes,
            'image_file': img_file
        }
def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

def evaluate_model(model, dataset, iou_threshold=0.5, conf_threshold=0.3):
    """Evaluate the model on the dataset and return performance metrics."""
    true_positives, false_positives, false_negatives = 0, 0, 0
    matches = []
    confidences = []

    for data in tqdm(dataset, desc="Evaluating Model"):
        image = data['image']
        gt_boxes = data['boxes']
        
        # Run YOLO inference
        results = model.predict(image, verbose=False)
        
        pred_boxes = []
        pred_confs = []
        
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                if conf >= conf_threshold:
                    pred_boxes.append(box.cpu().numpy())
                    pred_confs.append(conf.item())
        
        # Match predictions with ground truth
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        for pred_box, conf in zip(pred_boxes, pred_confs):
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                true_positives += 1
                gt_matched[best_gt_idx] = True
                matches.append(1)
            else:
                false_positives += 1
                matches.append(0)
            
            confidences.append(conf)
        
        false_negatives += np.sum(~gt_matched)

    # Compute Precision, Recall, F1-score
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": matches,
        "confidences": confidences
    }

# [Previous evaluate_model and calculate_iou functions remain the same]

def plot_metrics(results, model_name):
    """Plot various performance metrics"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    
    # Add timestamp and model name as figure suptitle
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.suptitle(f'Model: {model_name}\nTimestamp: {timestamp}', y=1.02)
    
    # 1. Basic Metrics Bar Plot
    plt.subplot(2, 2, 1)
    metrics = results['metrics']
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.bar(names, values)
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

    # 2. Precision-Recall Curve
    plt.subplot(2, 2, 2)
    matches = np.array(results['matches'])
    confidences = np.array(results['confidences'])
    if len(matches) > 0 and len(confidences) > 0:
        precision, recall, _ = precision_recall_curve(matches, confidences)
        plt.plot(recall, precision)
        plt.fill_between(recall, precision, alpha=0.2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve\nAP: {average_precision_score(matches, confidences):.3f}')

    # 3. Confusion Matrix Heatmap
    plt.subplot(2, 2, 3)
    conf_matrix = np.array([
        [results['true_positives'], results['false_positives']],
        [results['false_negatives'], 0]  # We don't have true negatives
    ])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'])
    plt.title('Confusion Matrix')
    
    # 4. Additional Information
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = (
        f"Model Performance Summary:\n\n"
        f"Total Predictions: {len(results['matches'])}\n"
        f"True Positives: {results['true_positives']}\n"
        f"False Positives: {results['false_positives']}\n"
        f"False Negatives: {results['false_negatives']}\n\n"
        f"Precision: {results['metrics']['precision']:.4f}\n"
        f"Recall: {results['metrics']['recall']:.4f}\n"
        f"F1 Score: {results['metrics']['f1_score']:.4f}\n"
    )
    plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    image_dir = "/home/ozgur/Ozgur_File/OzgursModel/Dataset/Kitty_DataSet/image/training/image_2"
    label_dir = "/home/ozgur/Ozgur_File/OzgursModel/Dataset/Kitty_DataSet/label/training/label_2"
    performance_dir = "/home/ozgur/Ozgur_File/OzgursModel/Performance"
    
    # Create performance directory if it doesn't exist
    os.makedirs(performance_dir, exist_ok=True)
    
    # Create dataset
    dataset = KITTIDataset(image_dir, label_dir)
    
    # Load YOLO model and get model name
    model_path = "/home/ozgur/Ozgur_File/OzgursModel/Models/yolov11n.pt"
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_model(model, dataset)
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    
    # Plot metrics and save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_fig = plot_metrics(results, model_name)
    
    # Save figure as SVG
    save_path = os.path.join(performance_dir, f"Performance_{timestamp}.svg")
    metrics_fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\nPerformance metrics saved to: {save_path}")
    
    # Display plot
    plt.show()