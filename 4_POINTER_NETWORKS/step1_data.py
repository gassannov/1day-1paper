import numpy as np
from scipy.spatial import ConvexHull
import torch


class ConvexHullDataset:
    def __init__(self, num_samples, num_points_range=(5, 50)):
        self.num_samples = num_samples
        self.num_points_range = num_points_range
        self.data = []
        self._generate()
    
    def _generate(self):
        for _ in range(self.num_samples):
            n = np.random.randint(*self.num_points_range)
            points = np.random.uniform(0, 1, size=(n, 2))

            hull = ConvexHull(points)
            hull_indices = hull.vertices

            start_idx = np.argmin(hull_indices)
            hull_indices = np.roll(hull_indices, -start_idx)
            
            self.data.append({
                'points': points.astype(np.float32),
                'hull': hull_indices.astype(np.int64)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    batch_size = len(batch)
    max_len = max(item['points'].shape[0] for item in batch)
    max_hull_len = max(item['hull'].shape[0] for item in batch)
    
    points_padded = np.zeros((batch_size, max_len, 2), dtype=np.float32)
    hull_padded = np.zeros((batch_size, max_hull_len + 2), dtype=np.int64)
    points_len = np.zeros(batch_size, dtype=np.int64)
    hull_len = np.zeros(batch_size, dtype=np.int64)
    
    for i, item in enumerate(batch):
        n = item['points'].shape[0]
        h = item['hull'].shape[0]
        
        points_padded[i, :n] = item['points']
        hull_padded[i, 0] = 0  # start token
        hull_padded[i, 1:h+1] = item['hull'] + 1  # +1 for start token offset
        hull_padded[i, h+1] = 0  # end token (same as start)
        
        points_len[i] = n
        hull_len[i] = h + 2
    
    return {
        'points': torch.FloatTensor(points_padded),
        'hull': torch.LongTensor(hull_padded),
        'points_len': torch.LongTensor(points_len),
        'hull_len': torch.LongTensor(hull_len)
    }


if __name__ == '__main__':
    dataset = ConvexHullDataset(num_samples=10, num_points_range=(5, 20))
    
    print(f"Generated {len(dataset)} samples")
    print(f"\nSample 0:")
    sample = dataset[0]
    print(f"Points shape: {sample['points'].shape}")
    print(f"Hull indices: {sample['hull']}")
    print(f"Points:\n{sample['points'][:5]}")
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"\nBatch shapes:")
    print(f"Points: {batch['points'].shape}")
    print(f"Hull: {batch['hull'].shape}")
    print(f"Points lengths: {batch['points_len']}")
    print(f"Hull lengths: {batch['hull_len']}")
