"""
LSTM (Long Short-Term Memory) Character-Level Language Model

LSTMs are better than vanilla RNNs because they can:
1. Remember information for longer periods
2. Decide what to forget and what to remember
3. Handle vanishing gradient problem better

LSTM has 4 gates:
- Forget gate: decides what to throw away from cell state
- Input gate: decides what new information to store
- Cell state: the memory of the network
- Output gate: decides what to output
"""

import numpy as np
import matplotlib.pyplot as plt


class LSTM:
    """
    LSTM for character-level language modeling

    The key difference from vanilla RNN:
    - Vanilla RNN: h_t = tanh(W * [h_{t-1}, x_t])
    - LSTM: has gates that control information flow
    """

    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=1e-1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # LSTM has 4x the parameters of vanilla RNN!
        # Each gate (forget, input, candidate, output) has its own weights

        # Weights for: [forget, input, candidate, output] gates
        # All concatenated together for efficiency
        self.Wf = np.random.randn(hidden_size, vocab_size + hidden_size) * 0.01  # Forget gate
        self.Wi = np.random.randn(hidden_size, vocab_size + hidden_size) * 0.01  # Input gate
        self.Wc = np.random.randn(hidden_size, vocab_size + hidden_size) * 0.01  # Candidate
        self.Wo = np.random.randn(hidden_size, vocab_size + hidden_size) * 0.01  # Output gate

        # Biases
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        # Output layer (same as vanilla RNN)
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.by = np.zeros((vocab_size, 1))

        # Adagrad memory
        self.memory = {}
        for param_name in ['Wf', 'Wi', 'Wc', 'Wo', 'bf', 'bi', 'bc', 'bo', 'Why', 'by']:
            self.memory[param_name] = np.zeros_like(getattr(self, param_name))

        self.smooth_loss = -np.log(1.0/vocab_size) * seq_length

    def sigmoid(self, x):
        """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-x))

    def forward_step(self, x, h_prev, c_prev):
        """
        Single LSTM forward step

        LSTM equations:
        1. Forget gate:    f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)
        2. Input gate:     i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)
        3. Candidate:      c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)
        4. Cell state:     c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        5. Output gate:    o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)
        6. Hidden state:   h_t = o_t ⊙ tanh(c_t)

        ⊙ means element-wise multiplication
        """
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x))

        # 1. FORGET GATE: decides what to forget from cell state
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # 2. INPUT GATE: decides what new information to store
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)

        # 3. CANDIDATE: new candidate values to add to cell state
        c_candidate = np.tanh(np.dot(self.Wc, concat) + self.bc)

        # 4. CELL STATE: combine forget and input
        c = f * c_prev + i * c_candidate

        # 5. OUTPUT GATE: decides what to output
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

        # 6. HIDDEN STATE: filtered cell state
        h = o * np.tanh(c)

        # Cache for backward pass
        cache = (x, h_prev, c_prev, f, i, c_candidate, c, o, h, concat)

        return h, c, cache

    def forward_pass(self, inputs, h_prev, c_prev):
        """Forward pass through entire sequence"""
        xs, hs, cs, caches = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)

        ps = {}  # Probability distributions

        for t, input_idx in enumerate(inputs):
            # One-hot encode input
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][input_idx] = 1

            # LSTM step
            hs[t], cs[t], caches[t] = self.forward_step(xs[t], hs[t-1], cs[t-1])

            # Output layer
            y = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(y) / np.sum(np.exp(y))

        return xs, hs, cs, ps, caches

    def backward_step(self, dh_next, dc_next, cache):
        """
        Single LSTM backward step (complex but powerful!)
        """
        x, h_prev, c_prev, f, i, c_candidate, c, o, h, concat = cache

        # Gradient from output
        do = dh_next * np.tanh(c)
        do = do * o * (1 - o)  # sigmoid derivative

        # Gradient flowing to cell state
        dc = dh_next * o * (1 - np.tanh(c)**2) + dc_next

        # Gradient for forget gate
        df = dc * c_prev
        df = df * f * (1 - f)
        
        # Gradient for input gate
        di = dc * c_candidate
        di = di * i * (1 - i)
        
        # Gradient for candidate
        dc_candidate = dc * i
        dc_candidate = dc_candidate * (1 - c_candidate**2)
        
        # Gradients for weights
        dWf = np.dot(df, concat.T)
        dWi = np.dot(di, concat.T)
        dWc = np.dot(dc_candidate, concat.T)
        dWo = np.dot(do, concat.T)
        
        dbf = df
        dbi = di
        dbc = dc_candidate
        dbo = do
        
        # Gradient flowing to previous states
        dconcat = (np.dot(self.Wf.T, df) + 
                   np.dot(self.Wi.T, di) + 
                   np.dot(self.Wc.T, dc_candidate) + 
                   np.dot(self.Wo.T, do))
        
        dh_prev = dconcat[:self.hidden_size, :]
        dc_prev = f * dc
        
        return dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dh_prev, dc_prev
    
    def compute_loss(self, ps, targets):
        """Cross-entropy loss"""
        loss = 0
        for t, target in enumerate(targets):
            loss += -np.log(ps[t][target, 0] + 1e-8)
        return loss
    
    def train_step(self, inputs, targets, h_prev, c_prev):
        """One training step"""
        # Forward
        xs, hs, cs, ps, caches = self.forward_pass(inputs, h_prev, c_prev)
        loss = self.compute_loss(ps, targets)
        
        # Backward
        gradients = {
            'Wf': np.zeros_like(self.Wf),
            'Wi': np.zeros_like(self.Wi),
            'Wc': np.zeros_like(self.Wc),
            'Wo': np.zeros_like(self.Wo),
            'bf': np.zeros_like(self.bf),
            'bi': np.zeros_like(self.bi),
            'bc': np.zeros_like(self.bc),
            'bo': np.zeros_like(self.bo),
            'Why': np.zeros_like(self.Why),
            'by': np.zeros_like(self.by)
        }
        
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        
        # Backprop through time
        for t in reversed(range(len(inputs))):
            # Gradient at output
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            
            gradients['Why'] += np.dot(dy, hs[t].T)
            gradients['by'] += dy
            
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # Backprop through LSTM
            dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dh_next, dc_next = \
                self.backward_step(dh, dc_next, caches[t])
            
            gradients['Wf'] += dWf
            gradients['Wi'] += dWi
            gradients['Wc'] += dWc
            gradients['Wo'] += dWo
            gradients['bf'] += dbf
            gradients['bi'] += dbi
            gradients['bc'] += dbc
            gradients['bo'] += dbo
        
        # Clip gradients
        for key in gradients:
            np.clip(gradients[key], -5, 5, out=gradients[key])
        
        # Update with Adagrad
        for param_name, grad in gradients.items():
            self.memory[param_name] += grad * grad
            param = getattr(self, param_name)
            param -= self.learning_rate * grad / np.sqrt(self.memory[param_name] + 1e-8)
        
        self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
        
        return loss, hs[len(inputs)-1], cs[len(inputs)-1]
    
    def sample(self, h, c, seed_ix, n, temperature=1.0):
        """Sample from LSTM"""
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        indices = []
        
        for _ in range(n):
            h, c, _ = self.forward_step(x, h, c)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
            
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            indices.append(ix)
        
        return indices


def train_lstm_on_text(text_file='shakespeare.txt', iterations=2000):
    """Train LSTM on text file"""
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"File {text_file} not found. Using sample text.")
        data = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles""" * 10
    
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Data size: {len(data)} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Characters: {chars}\n")
    
    # Create LSTM
    lstm = LSTM(
        vocab_size=vocab_size,
        hidden_size=128,
        seq_length=50,
        learning_rate=0.01
    )
    
    # Training loop
    p = 0
    h = np.zeros((lstm.hidden_size, 1))
    c = np.zeros((lstm.hidden_size, 1))
    losses = []

    print("Training LSTM...")
    for iteration in range(iterations):
        if p + lstm.seq_length + 1 >= len(data):
            h = np.zeros((lstm.hidden_size, 1))
            c = np.zeros((lstm.hidden_size, 1))
            p = 0

        inputs = [char_to_ix[ch] for ch in data[p:p+lstm.seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+lstm.seq_length+1]]

        loss, h, c = lstm.train_step(inputs, targets, h, c)
        losses.append(lstm.smooth_loss)

        if iteration % 100 == 0:
            sample_ix = lstm.sample(h, c, inputs[0], 200, temperature=0.8)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f"\nIter {iteration}, loss: {lstm.smooth_loss:.4f}")
            print(f"Sample:\n{txt[:150]}")

        p += lstm.seq_length

    # Final samples at different temperatures
    print("\n" + "="*60)
    print("FINAL SAMPLES AT DIFFERENT TEMPERATURES")
    print("="*60)

    for temp in [0.5, 1.0, 1.5]:
        sample_ix = lstm.sample(h, c, char_to_ix[data[0]], 300, temperature=temp)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f"\n--- Temperature {temp} ---")
        print(txt[:200])

    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Smooth Loss')
    plt.title('LSTM Training Loss')
    plt.grid(True)
    plt.savefig('/home/claude/lstm_loss.png', dpi=100, bbox_inches='tight')
    print("\nLoss plot saved to lstm_loss.png")

    return lstm, char_to_ix, ix_to_char


if __name__ == "__main__":
    train_lstm_on_text(iterations=1000)
