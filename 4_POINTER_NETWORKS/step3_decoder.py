import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_input, hidden, cell, mask=None):
        """
        Args:
            encoder_outputs: (batch, seq_len, hidden_dim)
            decoder_input: (batch, 1, input_dim)
            hidden: (1, batch, hidden_dim)
            cell: (1, batch, hidden_dim)
            mask: (batch, seq_len) - optional mask for invalid positions
        
        Returns:
            probs: (batch, seq_len) - probability distribution over input positions
            hidden: (1, batch, hidden_dim)
            cell: (1, batch, hidden_dim)
        """
        _, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
        
        decoder_hidden = hidden.squeeze(0)  # (batch, hidden_dim)
        
        encoder_transform = self.W1(encoder_outputs)  # (batch, seq_len, hidden_dim)
        decoder_transform = self.W2(decoder_hidden).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        tanh_sum = torch.tanh(encoder_transform + decoder_transform)  # (batch, seq_len, hidden_dim)
        
        scores = self.v(tanh_sum).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        probs = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        return probs, hidden, cell


if __name__ == '__main__':
    from step1_data import ConvexHullDataset, collate_fn
    from step2_encoder import Encoder
    from torch.utils.data import DataLoader
    
    dataset = ConvexHullDataset(num_samples=10, num_points_range=(5, 20))
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    encoder = Encoder(input_dim=2, hidden_dim=128)
    decoder = PointerDecoder(input_dim=2, hidden_dim=128)
    
    encoder_outputs, (hidden, cell) = encoder(batch['points'], batch['points_len'])
    
    start_input = torch.zeros(batch['points'].size(0), 1, 2)
    
    mask = torch.zeros(batch['points'].size(0), batch['points'].size(1))
    for i, length in enumerate(batch['points_len']):
        mask[i, :length] = 1
    
    probs, hidden, cell = decoder(encoder_outputs, start_input, hidden, cell, mask)
    
    print(f"Decoder input shape: {start_input.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs.sum(dim=1)}")
    print(f"\nFirst sample probabilities (should only be non-zero for valid positions):")
    print(probs[0])
    print(f"First sample has {batch['points_len'][0]} valid points")
