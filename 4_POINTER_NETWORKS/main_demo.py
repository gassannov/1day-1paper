"""
Pointer Networks - Complete Demo
=================================

This script demonstrates the complete workflow of training and evaluating
a Pointer Network on the Convex Hull problem.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from step1_data import ConvexHullDataset, collate_fn
from step4_model import PointerNetwork
from step5_train import train_epoch, evaluate, compute_loss
from step7_visualization import visualize_convex_hull, evaluate_model_detailed


def quick_demo():
    """Quick demonstration of the model"""
    print("=" * 70)
    print("POINTER NETWORKS - QUICK DEMO")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\n[1/5] Creating dataset...")
    train_dataset = ConvexHullDataset(num_samples=1000, num_points_range=(5, 20))
    val_dataset = ConvexHullDataset(num_samples=100, num_points_range=(5, 20))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    print("\n[2/5] Creating model...")
    model = PointerNetwork(input_dim=2, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    print("\n[3/5] Training model...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        print(f"   Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_accuracy:.4f}")
    
    print("\n[4/5] Visualizing predictions...")
    test_dataset = ConvexHullDataset(num_samples=10, num_points_range=(10, 15))
    
    for i in range(3):
        sample = test_dataset[i]
        points = torch.FloatTensor(sample['points']).unsqueeze(0).to(device)
        points_len = torch.LongTensor([len(sample['points'])]).to(device)
        
        model.eval()
        with torch.no_grad():
            pred_raw = model.inference(points, points_len)
            prediction = pred_raw[0].cpu().numpy()
            
            end_idx = np.where(prediction == 0)[0]
            if len(end_idx) > 0:
                prediction = prediction[:end_idx[0]]
            prediction = [p - 1 for p in prediction if p > 0]
        
        fig = visualize_convex_hull(
            sample['points'],
            prediction,
            sample['hull'],
            title=f"Example {i+1}"
        )
        plt.savefig(f'demo_prediction_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved demo_prediction_{i}.png")
    
    print("\n[5/5] Detailed evaluation...")
    results = evaluate_model_detailed(model.to('cpu'), test_dataset, num_samples=5, use_beam_search=False)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check generated visualizations (demo_prediction_*.png)")
    print("  2. Try training longer with more data")
    print("  3. Experiment with beam search")
    print("  4. Work through exercises.py")
    

def architecture_demo():
    """Demonstrate the architecture components separately"""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPONENTS DEMO")
    print("=" * 70)
    
    from step2_encoder import Encoder
    from step3_decoder import PointerDecoder
    
    batch_size = 2
    seq_len = 10
    input_dim = 2
    hidden_dim = 64
    
    points = torch.randn(batch_size, seq_len, input_dim)
    points_len = torch.LongTensor([10, 7])
    
    print("\n1. Encoder")
    print("   Input: (batch_size, seq_len, input_dim)")
    encoder = Encoder(input_dim, hidden_dim)
    encoder_outputs, (hidden, cell) = encoder(points, points_len)
    print(f"   Output: {encoder_outputs.shape}")
    print(f"   Hidden: {hidden.shape}, Cell: {cell.shape}")
    
    print("\n2. Decoder (Pointer Mechanism)")
    print("   Key innovation: attention becomes output distribution!")
    decoder = PointerDecoder(input_dim, hidden_dim)
    
    decoder_input = torch.zeros(batch_size, 1, input_dim)
    mask = torch.ones(batch_size, seq_len)
    
    probs, hidden, cell = decoder(encoder_outputs, decoder_input, hidden, cell, mask)
    print(f"   Output probabilities: {probs.shape}")
    print(f"   Probs sum to 1: {probs.sum(dim=1)}")
    print(f"   Example distribution: {probs[0]}")
    
    print("\n3. The Pointer Mechanism Math")
    print("   u_i^j = v^T * tanh(W1*e_j + W2*d_i)")
    print("   p(C_i | ...) = softmax(u_i)")
    print("   ")
    print("   This gives us a probability distribution over INPUT positions!")
    print("   No separate output vocabulary needed!")


def comparison_demo():
    """Compare Pointer Network with baseline approaches"""
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    
    print("\nProblem: Predict convex hull of variable-length point sets")
    print("\nApproach 1: Fixed vocabulary seq2seq")
    print("  ❌ Requires training separate model for each input length")
    print("  ❌ Cannot handle variable output dictionary size")
    
    print("\nApproach 2: Seq2seq with attention")
    print("  ❌ Still uses fixed output vocabulary")
    print("  ✓ Better than vanilla seq2seq")
    
    print("\nApproach 3: Pointer Network (This paper)")
    print("  ✓ Single model for all input lengths")
    print("  ✓ Output dictionary size = input length")
    print("  ✓ Generalizes to lengths not seen during training")
    print("  ✓ Clean and elegant solution")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'architecture':
            architecture_demo()
        elif sys.argv[1] == 'comparison':
            comparison_demo()
        elif sys.argv[1] == 'quick':
            quick_demo()
        else:
            print("Usage: python main_demo.py [quick|architecture|comparison]")
    else:
        architecture_demo()
        comparison_demo()
        quick_demo()
