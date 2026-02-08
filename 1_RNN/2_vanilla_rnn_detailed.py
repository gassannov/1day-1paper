"""
Detailed Vanilla RNN Implementation
This version has more explanation and visualization to understand what's happening
"""

import numpy as np
import matplotlib.pyplot as plt


class VanillaRNN:
    """
    A simple Vanilla RNN for character-level language modeling
    
    The RNN maintains a hidden state that gets updated at each time step.
    Core equation: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
    """
    
    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=1e-1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # Xavier initialization: scale by sqrt(1/n) where n is input size
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
        # Memory for Adagrad (adaptive learning rate)
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
        self.smooth_loss = -np.log(1.0/vocab_size) * seq_length
        
    def forward_pass(self, inputs, hprev):
        """
        Forward pass through the RNN
        
        Visualize this as: 
        x_0 -> [h_0] -> y_0
        x_1 -> [h_1] -> y_1
        x_2 -> [h_2] -> y_2
        
        Where each [h_t] depends on x_t and h_{t-1}
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        
        for t, input_idx in enumerate(inputs):
            # 1. Convert character index to one-hot vector
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][input_idx] = 1
            
            # 2. Update hidden state (this is the RNN magic!)
            # h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) +      # Current input contribution
                np.dot(self.Whh, hs[t-1]) +    # Previous hidden state contribution
                self.bh                         # Bias
            )
            
            # 3. Compute output logits
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            
            # 4. Apply softmax to get probability distribution
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        return xs, hs, ys, ps
    
    def backward_pass(self, xs, hs, ps, targets):
        """
        Backward pass (Backpropagation Through Time - BPTT)
        
        We go backwards through time, computing gradients
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros((self.hidden_size, 1))
        
        # Go backwards through time
        for t in reversed(range(len(targets))):
            # 1. Gradient at output layer
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # Cross-entropy gradient
            
            # 2. Accumulate gradients for Why and by
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            # 3. Backprop into hidden state
            dh = np.dot(self.Why.T, dy) + dhnext
            
            # 4. Backprop through tanh nonlinearity
            # tanh'(x) = 1 - tanh(x)^2
            dhraw = (1 - hs[t] * hs[t]) * dh
            
            # 5. Accumulate gradients for Wxh, Whh, bh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            
            # 6. Gradient flowing to previous time step
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # Clip gradients to prevent explosion
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def compute_loss(self, ps, targets):
        """Compute cross-entropy loss"""
        loss = 0
        for t, target in enumerate(targets):
            loss += -np.log(ps[t][target, 0])
        return loss
    
    def train_step(self, inputs, targets, hprev):
        """One training step"""
        # Forward pass
        xs, hs, ys, ps = self.forward_pass(inputs, hprev)
        
        # Compute loss
        loss = self.compute_loss(ps, targets)
        
        # Backward pass
        dWxh, dWhh, dWhy, dbh, dby = self.backward_pass(xs, hs, ps, targets)
        
        # Update weights using Adagrad
        for param, dparam, mem in [
            (self.Wxh, dWxh, self.mWxh),
            (self.Whh, dWhh, self.mWhh),
            (self.Why, dWhy, self.mWhy),
            (self.bh, dbh, self.mbh),
            (self.by, dby, self.mby)
        ]:
            mem += dparam * dparam
            param -= self.learning_rate * dparam / np.sqrt(mem + 1e-8)
        
        # Update smooth loss
        self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
        
        # Return last hidden state
        return loss, hs[len(inputs)-1]
    
    def sample(self, h, seed_ix, n, temperature=1.0):
        """
        Sample characters from the model
        
        Temperature controls randomness:
        - Low temperature (< 1): More confident, conservative
        - High temperature (> 1): More diverse, random
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        indices = []
        
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            # Apply temperature
            p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
            
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            indices.append(ix)
        
        return indices
    
    def get_hidden_state_visualization(self, text, char_to_ix):
        """
        Visualize how hidden states evolve as we process text
        Returns hidden state activations over time
        """
        inputs = [char_to_ix[ch] for ch in text]
        h = np.zeros((self.hidden_size, 1))
        hidden_states = []
        
        for input_idx in inputs:
            x = np.zeros((self.vocab_size, 1))
            x[input_idx] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hidden_states.append(h.ravel())
        
        return np.array(hidden_states)


def demonstrate_rnn():
    """Demonstrate RNN with a simple example"""
    
    # Simple toy data
    data = "hello world! this is a simple example. hello again!"
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Vocabulary: {chars}")
    print(f"Data: {data}")
    print(f"Vocab size: {vocab_size}\n")
    
    # Create RNN
    rnn = VanillaRNN(
        vocab_size=vocab_size,
        hidden_size=50,
        seq_length=25,
        learning_rate=0.1
    )
    
    # Training
    n, p = 0, 0
    hprev = np.zeros((rnn.hidden_size, 1))
    losses = []
    
    print("Training...")
    for iteration in range(1000):
        # Reset if at end of data
        if p + rnn.seq_length + 1 >= len(data):
            hprev = np.zeros((rnn.hidden_size, 1))
            p = 0
        
        # Get batch
        inputs = [char_to_ix[ch] for ch in data[p:p+rnn.seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+rnn.seq_length+1]]
        
        # Train
        loss, hprev = rnn.train_step(inputs, targets, hprev)
        losses.append(rnn.smooth_loss)
        
        # Sample periodically
        if iteration % 100 == 0:
            sample_ix = rnn.sample(hprev, inputs[0], 100, temperature=0.8)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f"\nIteration {iteration}, loss: {rnn.smooth_loss:.4f}")
            print(f"Sample: {txt[:60]}...")
        
        p += rnn.seq_length
    
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('rnn_loss.png', dpi=100, bbox_inches='tight')
    print("\nLoss plot saved to rnn_loss.png")
    
    # Generate samples at different temperatures
    print("\n" + "="*60)
    print("Samples at different temperatures:")
    print("="*60)
    
    for temp in [0.5, 1.0, 1.5]:
        sample_ix = rnn.sample(hprev, char_to_ix['h'], 100, temperature=temp)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f"\nTemperature {temp}:")
        print(txt[:80])


if __name__ == "__main__":
    demonstrate_rnn()
