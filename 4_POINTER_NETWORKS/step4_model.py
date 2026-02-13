import torch
import torch.nn as nn
from step2_encoder import Encoder
from step3_decoder import PointerDecoder


class PointerNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = PointerDecoder(input_dim, hidden_dim)
    
    def forward(self, points, target_indices, points_len):
        """
        Args:
            points: (batch, max_seq_len, input_dim)
            target_indices: (batch, max_target_len) - ground truth indices
            points_len: (batch,) - actual length of each sequence
        
        Returns:
            log_probs: (batch, max_target_len, max_seq_len)
        """
        batch_size = points.size(0)
        max_seq_len = points.size(1)
        max_target_len = target_indices.size(1)
        
        encoder_outputs, (hidden, cell) = self.encoder(points, points_len)
        
        mask = torch.zeros(batch_size, max_seq_len, device=points.device)
        for i, length in enumerate(points_len):
            mask[i, :length] = 1
        
        log_probs = []
        decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=points.device)
        
        for t in range(max_target_len - 1):
            probs, hidden, cell = self.decoder(
                encoder_outputs, decoder_input, hidden, cell, mask
            )
            log_probs.append(probs.log().unsqueeze(1))
            
            if self.training:
                current_indices = target_indices[:, t].unsqueeze(1).unsqueeze(2).expand(-1, -1, self.input_dim)
                current_indices = current_indices.clamp(0, max_seq_len - 1)
                decoder_input = torch.gather(points, 1, current_indices)
            else:
                selected_indices = probs.argmax(dim=1).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.input_dim)
                decoder_input = torch.gather(points, 1, selected_indices)
        
        log_probs = torch.cat(log_probs, dim=1)  # (batch, max_target_len-1, max_seq_len)
        
        return log_probs
    
    def inference(self, points, points_len, max_steps=50):
        """
        Greedy inference: select highest probability at each step
        """
        batch_size = points.size(0)
        max_seq_len = points.size(1)
        
        encoder_outputs, (hidden, cell) = self.encoder(points, points_len)
        
        mask = torch.zeros(batch_size, max_seq_len, device=points.device)
        for i, length in enumerate(points_len):
            mask[i, :length] = 1
        
        predictions = []
        decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=points.device)
        
        for t in range(max_steps):
            probs, hidden, cell = self.decoder(
                encoder_outputs, decoder_input, hidden, cell, mask
            )
            
            selected_indices = probs.argmax(dim=1)
            predictions.append(selected_indices.unsqueeze(1))
            
            if (selected_indices == 0).all():
                break
            
            selected_indices_expanded = selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.input_dim)
            decoder_input = torch.gather(points, 1, selected_indices_expanded)
        
        predictions = torch.cat(predictions, dim=1)  # (batch, steps)
        
        return predictions


if __name__ == '__main__':
    from step1_data import ConvexHullDataset, collate_fn
    from torch.utils.data import DataLoader
    
    dataset = ConvexHullDataset(num_samples=10, num_points_range=(5, 20))
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    model = PointerNetwork(input_dim=2, hidden_dim=128)
    model.train()
    
    log_probs = model(batch['points'], batch['hull'], batch['points_len'])
    
    print(f"Points shape: {batch['points'].shape}")
    print(f"Target hull shape: {batch['hull'].shape}")
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"\nGround truth hull for first sample: {batch['hull'][0]}")
    
    model.eval()
    with torch.no_grad():
        predictions = model.inference(batch['points'], batch['points_len'])
    
    print(f"Predicted hull for first sample: {predictions[0]}")
