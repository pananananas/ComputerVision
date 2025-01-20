from sklearn.metrics import precision_recall_curve
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
import torch

def create_heatmap(results, score_thresholds, nms_thresholds, plot_dir):
    # Create heatmap of results
    results_array = np.zeros((len(score_thresholds), len(nms_thresholds)))
    for r in results:
        i = score_thresholds.index(r['score_thresh'])
        j = nms_thresholds.index(r['nms_thresh'])
        results_array[i, j] = r['map']

    plt.figure(figsize=(10, 8))
    plt.imshow(results_array, cmap='viridis')
    plt.colorbar(label='mAP')
    plt.xticks(range(len(nms_thresholds)), nms_thresholds)
    plt.yticks(range(len(score_thresholds)), score_thresholds)
    plt.xlabel('NMS Threshold')
    plt.ylabel('Score Threshold')
    plt.title('mAP for Different Parameter Combinations')
    plt.savefig(plot_dir / 'parameter_heatmap.png')
    plt.close()

    print("\nText-based heatmap of mAP scores:")
    print("\nNMS thresh ->")
    print("Score     {:.1f}   {:.1f}   {:.1f}   {:.1f}   {:.1f}   {:.1f}   {:.1f}   {:.1f}   {:.1f}".format(*nms_thresholds))
    print("thresh")
    print("     v")
    for i, score_t in enumerate(score_thresholds):
        row = [results_array[i, j] for j in range(len(nms_thresholds))]
        print("{:.2f}    {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(score_t, *row))


def box_iou(box1, box2):

    # Get coordinates of intersection
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    
    # Calculate area of intersection
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate area of both boxes
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Calculate IoU
    union = box1_area + box2_area - intersection
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return iou


def get_precision_recall_data(model, val_loader, score_thresh, nms_thresh, device):

    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = nms_thresh
    
    all_scores = []
    all_correct = []
    
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            predictions = model(images)
            
            # For each image in batch
            for pred, target in zip(predictions, targets):
                pred_boxes = pred['boxes'].cpu()
                pred_scores = pred['scores'].cpu()
                pred_labels = pred['labels'].cpu()
                
                target_boxes = target['boxes'].cpu()
                target_labels = target['labels'].cpu()
                
                # For each prediction, check if it matches any ground truth
                for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
                    all_scores.append(pred_score.item())
                    
                    # Get matching ground truth boxes (same class)
                    matching_targets = target_boxes[target_labels == pred_label]
                    
                    if len(matching_targets) > 0:
                        # Calculate IoU with all matching ground truth boxes
                        ious = box_iou(pred_box.unsqueeze(0), matching_targets)
                        # Consider it correct if IoU > 0.5 with any matching ground truth
                        correct = (ious > 0.5).any().item()
                    else:
                        correct = False
                        
                    all_correct.append(correct)
    
    return np.array(all_scores), np.array(all_correct)


def plot_precision_recall_curves(model, val_loader, score_thresholds, nms_thresholds, device, save_dir):
    """Plot precision-recall curves for different parameter combinations"""
    plt.figure(figsize=(12, 8))
    
    for nms_t in nms_thresholds:
        for score_t in score_thresholds:
            scores, correct = get_precision_recall_data(model, val_loader, score_t, nms_t, device)
            
            if len(scores) > 0:  # Only plot if we have predictions
                precision, recall, _ = precision_recall_curve(correct, scores)
                plt.plot(recall, precision, 
                        label=f'NMS={nms_t}, Score={score_t}',
                        alpha=0.7)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Different Parameters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curves.png', bbox_inches='tight', dpi=300)
    plt.close()


def visualize_proposals_and_predictions(img, proposals, predictions, dataset, save_path):
    """Create side-by-side visualization of proposals and predictions"""
    # Draw proposals (green)
    proposals_vis = draw_bounding_boxes(
        img.clone(),
        boxes=proposals,
        colors='green',
        width=1
    )
    
    # Draw predictions (red) with scores and labels
    predictions_vis = draw_bounding_boxes(
        img.clone(),
        boxes=predictions['boxes'],
        labels=[f"{dataset.coco.cats[i.item()]['name']}\n{s:.2f}" 
                for i, s in zip(predictions['labels'], predictions['scores'])],
        colors='red',
        width=2
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(proposals_vis.permute(1, 2, 0))
    ax1.set_title(f'RPN Proposals\n({len(proposals)} proposals)', fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(predictions_vis.permute(1, 2, 0))
    ax2.set_title(f'Final Predictions\n({len(predictions["boxes"])} detections)', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_rpn_nms_analysis(results, plot_dir, print_results=True):
    plt.figure(figsize=(10, 6))
    plt.plot([r['rpn_nms_thresh'] for r in results], 
            [r['map'] for r in results], 
            marker='o')
    plt.xlabel('RPN NMS Threshold')
    plt.ylabel('mAP')
    plt.title('Effect of RPN NMS Threshold on Detection Performance')
    plt.grid(True)
    plt.savefig(plot_dir / 'rpn_nms_map.png')
    plt.close()

    if print_results:
        # Print results table
        print("\nResults Summary:")
        print("RPN NMS Threshold | mAP")
        print("-" * 30)
        for r in results:
            print(f"{r['rpn_nms_thresh']:15.2f} | {r['map']:.4f}")


def visualize_timing_analysis(timing_results, plot_dir, rpn_nms_thresholds):

    plt.figure(figsize=(12, 6))
    for metric in ['rpn_time', 'roi_time', 'total_time', 'conv_time']:
        avg_times = [np.mean([r[metric] for r in timing_results if r['rpn_nms_thresh'] == t]) 
                    for t in rpn_nms_thresholds]
        plt.plot(rpn_nms_thresholds, avg_times, marker='o', label=metric)
    
    plt.xlabel('RPN NMS Threshold')
    plt.ylabel('Time (seconds)')
    plt.title('Effect of RPN NMS Threshold on Processing Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / 'rpn_nms_timing.png')
    plt.close()


