import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, (hidden, cell)


if __name__ == '__main__':
    from step1_data import ConvexHullDataset, collate_fn
    from torch.utils.data import DataLoader
    
    dataset = ConvexHullDataset(num_samples=10, num_points_range=(5, 20))
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    encoder = Encoder(input_dim=2, hidden_dim=128)
    encoder_outputs, (hidden, cell) = encoder(batch['points'], batch['points_len'])
    
    print(f"Input shape: {batch['points'].shape}")
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    print(f"Cell state shape: {cell.shape}")
    
    print(f"\nFirst sample has {batch['points_len'][0]} points")
    print(f"Encoder output for first sample (first 3 timesteps):")
    print(encoder_outputs[0, :3, :5])
