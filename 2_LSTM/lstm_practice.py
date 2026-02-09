import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


print("=" * 60)
print("STEP 1: Simple RNN Cell")
print("=" * 60)


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        h = tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        return h


input_size = 3
hidden_size = 4
rnn_cell = SimpleRNNCell(input_size, hidden_size)

x = np.random.randn(input_size, 1)
h_prev = np.zeros((hidden_size, 1))
h = rnn_cell.forward(x, h_prev)

print(f"Input shape: {x.shape}")
print(f"Previous hidden state shape: {h_prev.shape}")
print(f"New hidden state shape: {h.shape}")
print(f"Hidden state values:\n{h}")
print("\n" + "=" * 60)
print("STEP 2: LSTM with Forget Gate Only")
print("=" * 60)


class LSTMWithForgetGate:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack([h_prev, x])
        f = sigmoid(np.dot(self.Wf, combined) + self.bf)
        c = f * c_prev
        h = tanh(c)
        return h, c, f


lstm_forget = LSTMWithForgetGate(input_size, hidden_size)

c_prev = np.random.randn(hidden_size, 1)
h, c, f = lstm_forget.forward(x, h_prev, c_prev)

print(f"Forget gate values:\n{f}")
print(f"\nCell state before: {c_prev.T}")
print(f"Cell state after: {c.T}")
print("\n" + "=" * 60)
print("STEP 3: LSTM with Forget + Input Gates")
print("=" * 60)


class LSTMWithInputGate:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack([h_prev, x])
        f = sigmoid(np.dot(self.Wf, combined) + self.bf)
        i = sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_tilde = tanh(np.dot(self.Wc, combined) + self.bc)
        c = f * c_prev + i * c_tilde
        h = tanh(c)
        return h, c, f, i, c_tilde


lstm_input = LSTMWithInputGate(input_size, hidden_size)
h, c, f, i, c_tilde = lstm_input.forward(x, h_prev, c_prev)

print(f"Forget gate: {f.T}")
print(f"Input gate: {i.T}")
print(f"Candidate values: {c_tilde.T}")
print(f"New cell state: {c.T}")
print("\n" + "=" * 60)
print("STEP 4: Complete LSTM with Output Gate")
print("=" * 60)


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack([h_prev, x])
        f = sigmoid(np.dot(self.Wf, combined) + self.bf)
        i = sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_tilde = tanh(np.dot(self.Wc, combined) + self.bc)
        c = f * c_prev + i * c_tilde
        o = sigmoid(np.dot(self.Wo, combined) + self.bo)
        h = o * tanh(c)
        return h, c, {'f': f, 'i': i, 'o': o, 'c_tilde': c_tilde}


lstm_cell = LSTMCell(input_size, hidden_size)
h, c, gates = lstm_cell.forward(x, h_prev, c_prev)

print(f"Forget gate: {gates['f'].T}")
print(f"Input gate: {gates['i'].T}")
print(f"Output gate: {gates['o'].T}")
print(f"Candidate values: {gates['c_tilde'].T}")
print(f"Cell state: {c.T}")
print(f"Hidden state: {h.T}")
print("\n" + "=" * 60)
print("STEP 5: LSTM Processing a Sequence")
print("=" * 60)


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequence):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        outputs = []
        states = []
        for t in range(sequence.shape[0]):
            x = sequence[t].reshape(-1, 1)
            h, c, gates = self.cell.forward(x, h, c)
            outputs.append(h)
            states.append({'h': h, 'c': c, 'gates': gates})
        return outputs, states


sequence_length = 5
sequence = np.random.randn(sequence_length, input_size)
lstm = LSTM(input_size, hidden_size)
outputs, states = lstm.forward(sequence)

print(f"Sequence length: {sequence_length}")
print(f"Input at each step: {input_size} features")
print(f"Hidden size: {hidden_size}")
print("\nHidden states over time:")

for t, output in enumerate(outputs):
    print(f"t={t}: {output.T}")

print("\n" + "=" * 60)
print("STEP 6: Visualizing Gate Activations")
print("=" * 60)

fig, axes = plt.subplots(4, 1, figsize=(10, 8))
gate_names = ['Forget', 'Input', 'Output', 'Cell State']
gate_keys = ['f', 'i', 'o']

for idx, (ax, name) in enumerate(zip(axes[:3], gate_names[:3])):
    gate_values = [states[t]['gates'][gate_keys[idx]].flatten() for t in range(sequence_length)]
    gate_array = np.array(gate_values).T
    im = ax.imshow(gate_array, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_ylabel(f'{name} Gate')
    ax.set_yticks(range(hidden_size))
    ax.set_xticks(range(sequence_length))
    plt.colorbar(im, ax=ax)

cell_values = [states[t]['c'].flatten() for t in range(sequence_length)]
cell_array = np.array(cell_values).T
im = axes[3].imshow(cell_array, aspect='auto', cmap='coolwarm')

axes[3].set_ylabel('Cell State')
axes[3].set_yticks(range(hidden_size))
axes[3].set_xticks(range(sequence_length))
axes[3].set_xlabel('Time Step')

plt.colorbar(im, ax=axes[3])
plt.tight_layout()
plt.savefig('2_LSTM/lstm_gates_visualization.png', dpi=100, bbox_inches='tight')

print("Visualization saved as 'lstm_gates_visualization.png'")
print("\n" + "=" * 60)
print("STEP 7: GRU (Gated Recurrent Unit) - Simplified LSTM")
print("=" * 60)


class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bz = np.zeros((hidden_size, 1))
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.br = np.zeros((hidden_size, 1))
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
    
    def forward(self, x, h_prev):
        combined = np.vstack([h_prev, x])
        z = sigmoid(np.dot(self.Wz, combined) + self.bz)
        r = sigmoid(np.dot(self.Wr, combined) + self.br)
        combined_reset = np.vstack([r * h_prev, x])
        h_tilde = tanh(np.dot(self.Wh, combined_reset) + self.bh)
        h = (1 - z) * h_prev + z * h_tilde
        return h, {'z': z, 'r': r, 'h_tilde': h_tilde}


gru_cell = GRUCell(input_size, hidden_size)
h, gru_gates = gru_cell.forward(x, h_prev)

print(f"Update gate (z): {gru_gates['z'].T}")
print(f"Reset gate (r): {gru_gates['r'].T}")
print(f"Candidate hidden (h_tilde): {gru_gates['h_tilde'].T}")
print(f"Final hidden state: {h.T}")
print("\n" + "=" * 60)
print("STEP 8: Character-Level Language Model with LSTM")
print("=" * 60)

text = "hello world"
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Text: '{text}'")
print(f"Vocabulary: {chars}")
print(f"Vocab size: {vocab_size}")


class CharLSTM:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCell(vocab_size, hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.by = np.zeros((vocab_size, 1))

    def forward(self, char_idx, h_prev, c_prev):
        x = np.zeros((self.vocab_size, 1))
        x[char_idx] = 1
        h, c, gates = self.lstm.forward(x, h_prev, c_prev)
        y = np.dot(self.Why, h) + self.by
        probs = softmax(y)
        return probs, h, c


char_lstm = CharLSTM(vocab_size, hidden_size=16)
h = np.zeros((16, 1))
c = np.zeros((16, 1))

print("\nProcessing 'hello':")
for char in "hello":
    idx = char_to_idx[char]
    probs, h, c = char_lstm.forward(idx, h, c)
    predicted_idx = np.argmax(probs)
    predicted_char = idx_to_char[predicted_idx]
    print(f"Input: '{char}' -> Predicted next: '{predicted_char}' (prob: {probs[predicted_idx, 0]:.3f})")

print("\n" + "=" * 60)
print("STEP 9: Demonstrating Long-Term Dependencies")
print("=" * 60)

long_sequence = np.random.randn(20, input_size)
long_sequence[0, 0] = 5.0
long_sequence[19, 0] = 5.0
outputs, states = lstm.forward(long_sequence)
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
cell_memory = np.array([s['c'][0, 0] for s in states])
hidden_memory = np.array([s['h'][0, 0] for s in states])

axes[0].plot(cell_memory, 'b-o', label='Cell State (c)')
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Signal at t=0')
axes[0].axvline(x=19, color='g', linestyle='--', alpha=0.5, label='Signal at t=19')
axes[0].set_ylabel('Cell State Value')
axes[0].set_title('LSTM Cell State - Long-Term Memory')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(hidden_memory, 'r-o', label='Hidden State (h)')
axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1].axvline(x=19, color='g', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Hidden State Value')
axes[1].set_title('LSTM Hidden State Output')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_LSTM/lstm_long_term_memory.png', dpi=100, bbox_inches='tight')

print("Long-term dependency visualization saved!")
print(f"Cell state at t=0: {states[0]['c'][0, 0]:.4f}")
print(f"Cell state at t=10: {states[10]['c'][0, 0]:.4f}")
print(f"Cell state at t=19: {states[19]['c'][0, 0]:.4f}")
print("\n" + "=" * 60)
print("STEP 10: Comparing RNN vs LSTM vs GRU")
print("=" * 60)


class GRU:
    def __init__(self, input_size, hidden_size):
        self.cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequence):
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        for t in range(sequence.shape[0]):
            x = sequence[t].reshape(-1, 1)
            h, gates = self.cell.forward(x, h)
            outputs.append(h)
        return outputs


class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.cell = SimpleRNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequence):
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        for t in range(sequence.shape[0]):
            x = sequence[t].reshape(-1, 1)
            h = self.cell.forward(x, h)
            outputs.append(h)
        return outputs


test_sequence = np.random.randn(10, input_size)

rnn = SimpleRNN(input_size, hidden_size)
lstm = LSTM(input_size, hidden_size)
gru = GRU(input_size, hidden_size)

rnn_outputs = rnn.forward(test_sequence)
lstm_outputs, _ = lstm.forward(test_sequence)
gru_outputs = gru.forward(test_sequence)

print("Model outputs at final timestep:")
print(f"RNN:  {rnn_outputs[-1].T}")
print(f"LSTM: {lstm_outputs[-1].T}")
print(f"GRU:  {gru_outputs[-1].T}")
print("\nParameter counts:")

lstm_params = 4 * hidden_size * (input_size + hidden_size + 1)
gru_params = 3 * hidden_size * (input_size + hidden_size + 1)
rnn_params = hidden_size * (input_size + hidden_size + 1)
print(f"RNN:  {rnn_params} parameters")
print(f"LSTM: {lstm_params} parameters")
print(f"GRU:  {gru_params} parameters")
print("\n" + "=" * 60)
print("PRACTICE COMPLETE!")

print("=" * 60)
print("\nKey Takeaways:")
print("1. LSTM has 4 gates: forget, input, candidate, output")
print("2. Cell state (c) carries long-term memory")
print("3. Hidden state (h) is the filtered output")
print("4. GRU simplifies LSTM by combining gates")
print("5. Gates use sigmoid (0-1) to control information flow")
print("6. tanh creates candidate values (-1 to 1)")
print("\nVisualization files created:")
print("- lstm_gates_visualization.png")
print("- lstm_long_term_memory.png")
