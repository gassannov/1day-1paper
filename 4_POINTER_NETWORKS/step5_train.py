import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from step1_data import ConvexHullDataset, collate_fn
from step4_model import PointerNetwork


def compute_loss(log_probs, target_indices, target_len):
    """
    Args:
        log_probs: (batch, max_target_len-1, max_seq_len)
        target_indices: (batch, max_target_len)
        target_len: (batch,)
    """
    batch_size = log_probs.size(0)
    max_target_len = target_indices.size(1)

    print(log_probs.shape, max_target_len-1, target_len[0].item()-1)
    
    loss = 0
    for i in range(batch_size):
        actual_len = target_len[i].item() - 1  # -1 because we don't predict after last token
        
        for t in range(actual_len):
            target_idx = target_indices[i, t + 1]  # +1 because first is start token
            log_prob = log_probs[i, t, target_idx]
            loss -= log_prob
    
    loss = loss / batch_size
    return loss


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        points = batch['points'].to(device)
        hull = batch['hull'].to(device)
        points_len = batch['points_len'].to(device)
        hull_len = batch['hull_len'].to(device)
        
        optimizer.zero_grad()
        
        log_probs = model(points, hull, points_len)
        
        loss = compute_loss(log_probs, hull, hull_len)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            points = batch['points'].to(device)
            hull = batch['hull'].to(device)
            points_len = batch['points_len'].to(device)
            hull_len = batch['hull_len'].to(device)
            
            log_probs = model(points, hull, points_len)
            loss = compute_loss(log_probs, hull, hull_len)
            total_loss += loss.item()
            
            predictions = model.inference(points, points_len)
            
            for i in range(len(predictions)):
                pred_len = (predictions[i] == 0).nonzero(as_tuple=True)[0]
                pred_len = pred_len[0].item() if len(pred_len) > 0 else len(predictions[i])
                true_len = hull_len[i].item()
                
                if pred_len == true_len:
                    pred_hull = predictions[i][:pred_len]
                    true_hull = hull[i][1:true_len]  # skip start token
                    
                    pred_set = set(pred_hull.cpu().numpy())
                    true_set = set(true_hull.cpu().numpy())
                    
                    if pred_set == true_set:
                        correct += 1
                
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy


def train_model(num_epochs=20, batch_size=128, hidden_dim=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = ConvexHullDataset(num_samples=10000, num_points_range=(5, 50))
    val_dataset = ConvexHullDataset(num_samples=1000, num_points_range=(5, 50))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    model = PointerNetwork(input_dim=2, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'pointer_net_best.pt')
            print("Saved best model!")
    
    return model


if __name__ == '__main__':
    model = train_model(num_epochs=5, batch_size=32, hidden_dim=128)
