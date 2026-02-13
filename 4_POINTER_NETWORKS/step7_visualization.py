import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from step1_data import ConvexHullDataset, collate_fn
from step4_model import PointerNetwork
from step6_beam_search import BeamSearchDecoder


def visualize_convex_hull(points, predicted_hull, true_hull=None, title="Convex Hull"):
    """
    Visualize points and predicted/true convex hull
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=100, alpha=0.6, label='Points')
    
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='bottom')
    
    if predicted_hull is not None and len(predicted_hull) > 0:
        pred_points = points[predicted_hull]
        pred_polygon = Polygon(pred_points, fill=False, edgecolor='red', 
                              linewidth=2, label='Predicted Hull')
        ax.add_patch(pred_polygon)
    
    if true_hull is not None and len(true_hull) > 0:
        true_points = points[true_hull]
        true_polygon = Polygon(true_points, fill=False, edgecolor='green', 
                              linewidth=2, linestyle='--', label='True Hull')
        ax.add_patch(true_polygon)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def calculate_area(points, hull_indices):
    """Calculate area of polygon defined by hull_indices"""
    if len(hull_indices) < 3:
        return 0.0
    
    hull_points = points[hull_indices]
    
    x = hull_points[:, 0]
    y = hull_points[:, 1]
    
    area = 0.5 * abs(sum(x[i] * y[(i+1) % len(x)] - x[(i+1) % len(x)] * y[i] 
                         for i in range(len(x))))
    return area


def evaluate_model_detailed(model, dataset, num_samples=10, use_beam_search=False):
    """
    Detailed evaluation with visualization
    """
    model.eval()
    
    if use_beam_search:
        beam_decoder = BeamSearchDecoder(model, beam_width=5)
    
    results = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        points = torch.FloatTensor(sample['points']).unsqueeze(0)
        points_len = torch.LongTensor([len(sample['points'])])
        
        with torch.no_grad():
            if use_beam_search:
                prediction = beam_decoder.decode(points, points_len)
                prediction = [p for p in prediction if p != 0]
            else:
                pred_raw = model.inference(points, points_len)
                prediction = pred_raw[0].cpu().numpy()
                end_idx = np.where(prediction == 0)[0]
                if len(end_idx) > 0:
                    prediction = prediction[:end_idx[0]]
                prediction = [p - 1 for p in prediction if p > 0]  # adjust for start token offset
        
        true_hull = sample['hull']
        true_area = calculate_area(sample['points'], true_hull)
        
        if len(prediction) > 2:
            pred_area = calculate_area(sample['points'], prediction)
            area_coverage = pred_area / true_area if true_area > 0 else 0
        else:
            pred_area = 0
            area_coverage = 0
        
        pred_set = set(prediction)
        true_set = set(true_hull)
        accuracy = 1.0 if pred_set == true_set else 0.0
        
        results.append({
            'points': sample['points'],
            'prediction': prediction,
            'true_hull': true_hull,
            'accuracy': accuracy,
            'area_coverage': area_coverage,
            'true_area': true_area,
            'pred_area': pred_area
        })
        
        fig = visualize_convex_hull(
            sample['points'], 
            prediction, 
            true_hull,
            title=f"Sample {i}: Accuracy={accuracy:.2f}, Area Coverage={area_coverage:.3f}"
        )
        plt.savefig(f'hull_visualization_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_coverage = np.mean([r['area_coverage'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({'Beam Search' if use_beam_search else 'Greedy'})")
    print(f"{'='*60}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Area Coverage: {avg_coverage:.4f}")
    print(f"\nPer-sample results:")
    for i, r in enumerate(results):
        print(f"  Sample {i}: Acc={r['accuracy']:.2f}, "
              f"Coverage={r['area_coverage']:.3f}, "
              f"Pred_len={len(r['prediction'])}, True_len={len(r['true_hull'])}")
    
    return results


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    test_dataset = ConvexHullDataset(num_samples=20, num_points_range=(10, 30))
    
    model = PointerNetwork(input_dim=2, hidden_dim=256)
    
    print("Evaluating with greedy decoding...")
    greedy_results = evaluate_model_detailed(
        model, test_dataset, num_samples=5, use_beam_search=False
    )
    
    print("\n" + "="*60)
    print("Evaluating with beam search...")
    beam_results = evaluate_model_detailed(
        model, test_dataset, num_samples=5, use_beam_search=True
    )
