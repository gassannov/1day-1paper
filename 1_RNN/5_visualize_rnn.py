"""
Visualize RNN Internal Representations

This recreates the amazing visualizations from Karpathy's article showing:
1. Hidden state activations over time
2. What individual neurons are "looking for"
3. Prediction confidence at each step
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


class RNNVisualizer:
    """
    Visualize what's happening inside an RNN
    
    Similar to the visualizations in the article that show:
    - Which neurons fire for URLs
    - Which neurons detect brackets [[]]
    - Which neurons count position in sequences
    """
    
    def __init__(self, rnn, char_to_ix, ix_to_char):
        self.rnn = rnn
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char
        self.vocab_size = len(char_to_ix)
    
    def get_hidden_activations(self, text):
        """
        Get hidden state activations for a piece of text
        
        Returns:
            hidden_states: (seq_len, hidden_size) array
            predictions: (seq_len, vocab_size) array  
        """
        h = np.zeros((self.rnn.hidden_size, 1))
        hidden_states = []
        predictions = []
        
        for char in text:
            # Skip characters not in vocabulary
            if char not in self.char_to_ix:
                continue
            
            # One-hot encode
            x = np.zeros((self.vocab_size, 1))
            x[self.char_to_ix[char]] = 1
            
            # Forward step
            h = np.tanh(np.dot(self.rnn.Wxh, x) + 
                       np.dot(self.rnn.Whh, h) + self.rnn.bh)
            y = np.dot(self.rnn.Why, h) + self.rnn.by
            p = np.exp(y) / np.sum(np.exp(y))
            
            hidden_states.append(h.ravel())
            predictions.append(p.ravel())
        
        return np.array(hidden_states), np.array(predictions)
    
    def visualize_neuron_activations(self, text, neuron_indices=None, save_path=None):
        """
        Visualize how specific neurons activate over text
        
        This recreates the visualization showing neurons that:
        - Detect quotes
        - Count brackets
        - Track URL state
        """
        hidden_states, _ = self.get_hidden_activations(text)
        
        if neuron_indices is None:
            # Find interesting neurons (high variance)
            variances = np.var(hidden_states, axis=0)
            neuron_indices = np.argsort(variances)[-5:]  # Top 5 most varying
        
        # Create custom colormap (blue to green)
        colors = ['#0000ff', '#00ffff', '#00ff00']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        fig, axes = plt.subplots(len(neuron_indices) + 1, 1, 
                                figsize=(16, 2*len(neuron_indices)))
        
        # Plot text with character indices
        ax = axes[0]
        ax.text(0.5, 0.5, text, fontfamily='monospace', fontsize=10,
                ha='center', va='center', wrap=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Input Text", fontsize=12, fontweight='bold')
        
        # Plot each neuron
        for idx, neuron_idx in enumerate(neuron_indices):
            ax = axes[idx + 1]
            activations = hidden_states[:, neuron_idx]
            
            # Create colored background for each character
            for i, (char, activation) in enumerate(zip(text, activations)):
                color_val = (activation + 1) / 2  # Scale from [-1,1] to [0,1]
                color = cmap(color_val)
                
                # Draw character with background color
                rect = mpatches.Rectangle((i, 0), 1, 1, 
                                         facecolor=color, edgecolor='none')
                ax.add_patch(rect)
                
                # Add character text
                display_char = char if char != '\n' else '↵'
                ax.text(i + 0.5, 0.5, display_char, 
                       fontfamily='monospace', fontsize=8,
                       ha='center', va='center')
            
            ax.set_xlim(0, len(text))
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(f"Neuron {neuron_idx} (variance: {np.var(activations):.3f})",
                        fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved neuron activation visualization to {save_path}")
        plt.close()
    
    def visualize_predictions(self, text, top_k=5, save_path=None):
        """
        Visualize top-k predictions at each step
        
        Shows what the model thinks should come next
        """
        hidden_states, predictions = self.get_hidden_activations(text)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # For each position, show top-k predictions
        for i, (char, pred) in enumerate(zip(text, predictions)):
            # Get top-k predictions
            top_indices = np.argsort(pred)[-top_k:][::-1]
            
            # Draw input character
            display_char = char if char != '\n' else '↵'
            ax.text(i, top_k + 1, display_char, fontsize=10,
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue'))
            
            # Draw predictions
            for rank, idx in enumerate(top_indices):
                predicted_char = self.ix_to_char[idx]
                if predicted_char == '\n':
                    predicted_char = '↵'
                
                probability = pred[idx]
                
                # Color based on probability
                alpha = probability
                color = plt.cm.Reds(probability)
                
                y_pos = top_k - rank
                ax.text(i, y_pos, predicted_char, fontsize=8,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=alpha))
                
                # Show probability
                ax.text(i + 0.3, y_pos, f'{probability:.2f}', 
                       fontsize=6, ha='left', va='center', color='gray')
        
        ax.set_xlim(-0.5, len(text))
        ax.set_ylim(-0.5, top_k + 2)
        ax.set_xlabel('Character Position', fontsize=12)
        ax.set_ylabel('Top Predictions (darker = higher probability)', fontsize=12)
        ax.set_title('RNN Predictions at Each Time Step', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved predictions visualization to {save_path}")
        plt.close()
    
    def analyze_interesting_neurons(self, test_texts):
        """
        Find neurons that do interesting things
        
        For example:
        - Neurons that activate on quotes
        - Neurons that track bracket depth
        - Neurons that detect URLs
        """
        all_patterns = {}
        
        for description, text in test_texts.items():
            hidden_states, _ = self.get_hidden_activations(text)
            
            # Find neurons with interesting patterns
            for neuron_idx in range(hidden_states.shape[1]):
                activations = hidden_states[:, neuron_idx]
                
                # Check for various patterns
                variance = np.var(activations)
                mean_abs = np.mean(np.abs(activations))
                spikiness = np.sum(np.abs(np.diff(activations)))
                
                if neuron_idx not in all_patterns:
                    all_patterns[neuron_idx] = []
                
                all_patterns[neuron_idx].append({
                    'text': description,
                    'variance': variance,
                    'mean_abs': mean_abs,
                    'spikiness': spikiness,
                    'activation_pattern': activations
                })
        
        # Print interesting neurons
        print("\n" + "="*60)
        print("INTERESTING NEURON ANALYSIS")
        print("="*60)
        
        # Find high-variance neurons
        neuron_variances = [(i, np.mean([p['variance'] for p in patterns]))
                           for i, patterns in all_patterns.items()]
        neuron_variances.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 most variable neurons:")
        for i, (neuron_idx, avg_var) in enumerate(neuron_variances[:10]):
            print(f"  {i+1}. Neuron {neuron_idx}: variance = {avg_var:.4f}")
        
        return all_patterns
    
    def visualize_hidden_state_heatmap(self, text, save_path=None):
        """
        Show all hidden states as a heatmap
        
        Rows = neurons, Columns = time steps
        Color = activation value
        """
        hidden_states, _ = self.get_hidden_activations(text)
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot heatmap
        im = ax.imshow(hidden_states.T, aspect='auto', cmap='RdBu_r',
                      vmin=-1, vmax=1, interpolation='nearest')
        
        # Add character labels
        ax.set_xticks(range(len(text)))
        display_chars = [c if c != '\n' else '↵' for c in text]
        ax.set_xticklabels(display_chars, fontsize=8, fontfamily='monospace')
        
        ax.set_xlabel('Character Position', fontsize=12)
        ax.set_ylabel('Neuron Index', fontsize=12)
        ax.set_title('Hidden State Activations Over Time', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation', fontsize=12)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        plt.close()


def demo_visualization():
    """
    Demo the visualization tools
    """
    print("Creating demo RNN for visualization...")
    
    # Simple dataset
    data = """
    The quick brown fox jumps over the lazy dog.
    [[Article]] about neural networks and [[deep learning]].
    Visit http://www.example.com for more information.
    "Hello," said the cat. "How are you?"
    """ * 5
    
    chars = sorted(list(set(data)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create simple RNN
    class SimpleRNN:
        def __init__(self, vocab_size, hidden_size):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
            self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
            self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
            self.bh = np.zeros((hidden_size, 1))
            self.by = np.zeros((vocab_size, 1))
    
    rnn = SimpleRNN(len(chars), 100)
    
    # Create visualizer
    viz = RNNVisualizer(rnn, char_to_ix, ix_to_char)
    
    # Test texts
    test_text = "The [[quick]] fox at http://www.site.com said \"hello\""
    
    print("\nGenerating visualizations...")
    
    # 1. Neuron activations
    viz.visualize_neuron_activations(
        test_text,
        save_path='/home/claude/neuron_activations.png'
    )
    
    # 2. Predictions
    viz.visualize_predictions(
        test_text[:30],  # Shorter for readability
        save_path='/home/claude/predictions.png'
    )
    
    # 3. Hidden state heatmap
    viz.visualize_hidden_state_heatmap(
        test_text,
        save_path='/home/claude/hidden_states_heatmap.png'
    )
    
    # 4. Analyze interesting neurons
    test_texts = {
        'quotes': '"Hello" and "world"',
        'brackets': '[[item1]] and [[item2]]',
        'url': 'http://www.example.com/path',
        'normal': 'just some regular text'
    }
    
    patterns = viz.analyze_interesting_neurons(test_texts)
    
    print("\n✓ All visualizations complete!")
    print("  - neuron_activations.png")
    print("  - predictions.png")
    print("  - hidden_states_heatmap.png")


if __name__ == "__main__":
    demo_visualization()
