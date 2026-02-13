import torch
import torch.nn.functional as F
from step4_model import PointerNetwork


class BeamSearchDecoder:
    def __init__(self, model, beam_width=5, max_steps=50):
        self.model = model
        self.beam_width = beam_width
        self.max_steps = max_steps
    
    def decode(self, points, points_len):
        """
        Args:
            points: (1, seq_len, input_dim) - single sample
            points_len: (1,)
        
        Returns:
            best_sequence: list of indices
        """
        device = points.device
        seq_len = points.size(1)
        input_dim = points.size(2)
        
        encoder_outputs, (hidden, cell) = self.model.encoder(points, points_len)
        
        mask = torch.zeros(1, seq_len, device=device)
        mask[0, :points_len[0]] = 1
        
        beams = [{
            'sequence': [0],  # start with start token
            'score': 0.0,
            'hidden': hidden,
            'cell': cell,
            'decoder_input': torch.zeros(1, 1, input_dim, device=device)
        }]
        
        completed_beams = []
        
        for step in range(self.max_steps):
            candidates = []
            
            for beam in beams:
                if beam['sequence'][-1] == 0 and len(beam['sequence']) > 1:
                    completed_beams.append(beam)
                    continue
                
                probs, hidden, cell = self.model.decoder(
                    encoder_outputs, 
                    beam['decoder_input'],
                    beam['hidden'],
                    beam['cell'],
                    mask
                )
                
                log_probs = probs.log()
                
                top_probs, top_indices = log_probs.topk(self.beam_width, dim=1)
                
                for k in range(self.beam_width):
                    idx = top_indices[0, k].item()
                    prob = top_probs[0, k].item()
                    
                    new_sequence = beam['sequence'] + [idx]
                    new_score = beam['score'] + prob
                    
                    new_decoder_input = torch.zeros(1, 1, input_dim, device=device)
                    if idx < seq_len:
                        idx_tensor = torch.tensor([[[idx]]], device=device).expand(1, 1, input_dim)
                        new_decoder_input = torch.gather(points, 1, idx_tensor)
                    
                    candidates.append({
                        'sequence': new_sequence,
                        'score': new_score,
                        'hidden': hidden,
                        'cell': cell,
                        'decoder_input': new_decoder_input
                    })
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = candidates[:self.beam_width]
            
            if all(b['sequence'][-1] == 0 for b in beams):
                completed_beams.extend(beams)
                break
        
        all_beams = completed_beams + beams
        all_beams.sort(key=lambda x: x['score'] / len(x['sequence']), reverse=True)
        
        if all_beams:
            return all_beams[0]['sequence'][1:]  # remove start token
        else:
            return []


if __name__ == '__main__':
    from step1_data import ConvexHullDataset, collate_fn
    from torch.utils.data import DataLoader
    
    dataset = ConvexHullDataset(num_samples=5, num_points_range=(10, 15))
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    model = PointerNetwork(input_dim=2, hidden_dim=128)
    model.eval()
    
    beam_decoder = BeamSearchDecoder(model, beam_width=5)
    
    batch = next(iter(loader))
    
    print("Greedy decoding:")
    with torch.no_grad():
        greedy_pred = model.inference(batch['points'], batch['points_len'])
    print(f"Greedy prediction: {greedy_pred[0]}")
    
    print("\nBeam search decoding:")
    with torch.no_grad():
        beam_pred = beam_decoder.decode(batch['points'], batch['points_len'])
    print(f"Beam search prediction: {beam_pred}")
    
    print(f"\nGround truth: {batch['hull'][0]}")
