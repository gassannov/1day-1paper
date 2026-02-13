import torch
import torch.nn as nn
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ZarembaLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(ZarembaLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # The core LSTM unit (handles the recurrence)
        # Note: We do NOT use the built-in 'dropout' param here because 
        # we want to manually control where it is applied for this lesson.
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        # The Dropout layer (Vertical regularization only)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, state):
        """
        x: Input tensor for current timestep [batch_size, input_size]
        state: Tuple of (hidden_state, cell_state) from previous timestep
        """

        # KEY STEP: Apply dropout to the input (vertical connection)
        # This matches the paper's formula: D(h^{l-1}_t)
        x_dropped = self.dropout(x)

        # Pass the noisy input and the CLEAN previous state to the LSTM
        # The state (h_t-1, c_t-1) is NOT dropped out, preserving long-term memory.
        h_t, c_t = self.lstm(x_dropped, state)

        return h_t, (h_t, c_t)


class ZarembaLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout_prob):
        super(ZarembaLanguageModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 1. Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 2. Stacked LSTM Layers
        # We create a list of our custom ZarembaLSTMLayer cells
        self.layers = nn.ModuleList()

        # Input to first layer is embedding_dim
        self.layers.append(ZarembaLSTMLayer(embedding_dim, hidden_size, dropout_prob))

        # Input to subsequent layers is hidden_size
        for _ in range(num_layers - 1):
            self.layers.append(ZarembaLSTMLayer(hidden_size, hidden_size, dropout_prob))

        # 3. Output Decoder
        # Apply dropout before the final affine transform as well
        self.final_dropout = nn.Dropout(dropout_prob)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Tie weights (Optional, but common in Language Models)
        # self.decoder.weight = self.embedding.weight 

    def forward(self, input_seq, hidden_states):
        """
        input_seq: [seq_len, batch_size] containing word indices
        hidden_states: List of (h, c) tuples for each layer
        """
        seq_len, batch_size = input_seq.size()

        # Initialize outputs container
        outputs = []

        # Loop through time (The Recurrent part)
        for t in range(seq_len):

            # Get the word vector for this timestep
            x_t = self.embedding(input_seq[t])  # [batch_size, embedding_dim]

            # Loop through layers (The Deep part)
            for layer_idx, layer in enumerate(self.layers):
                # Get the state for this layer
                h_prev, c_prev = hidden_states[layer_idx]

                # Update layer: x_t flows UP through the layers
                # The layer applies dropout to x_t internally before using it
                h_t, (h_new, c_new) = layer(x_t, (h_prev, c_prev))

                # Update state for next timestep
                hidden_states[layer_idx] = (h_new, c_new)

                # The output of this layer is the input to the next layer
                x_t = h_t

            # After passing through all layers, x_t is the top hidden state
            outputs.append(x_t)

        # Stack outputs [seq_len, batch_size, hidden_size]
        output_tensor = torch.stack(outputs)

        # Flatten for the decoder
        output_flattened = output_tensor.view(-1, self.hidden_size)

        # Apply final dropout
        output_dropped = self.final_dropout(output_flattened)

        # Decode to vocabulary
        logits = self.decoder(output_dropped)

        return logits, hidden_states

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros for all layers
        return [(torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size))
                for _ in range(self.num_layers)]


# ---- DUMMY DATA ----

# --- Configuration ---
VOCAB_SIZE = 1000
EMBED_DIM = 200
HIDDEN_SIZE = 200
NUM_LAYERS = 2
DROPOUT_PROB = 0.5  # Paper suggests 0.5 for small models, higher for large ones
BATCH_SIZE = 20
SEQ_LEN = 30

# --- Instantiate Model ---
model = ZarembaLanguageModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0) # Paper used SGD with high LR

# --- Dummy Data Generator ---
def get_batch():
    # Random integers representing words
    data = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE))
    targets = torch.randint(0, VOCAB_SIZE, (SEQ_LEN * BATCH_SIZE,))
    return data, targets


# --- Training Step ---
model.train() # Enable Dropout
hidden = model.init_hidden(BATCH_SIZE)

print("Starting dummy training step...")

# Forward pass
inputs, targets = get_batch()

# Detach hidden states to prevent backpropagating through the entire history
# (Truncated Backpropagation Through Time)
hidden = [(h.detach(), c.detach()) for h, c in hidden]

logits, new_hidden = model(inputs, hidden)

# Compute Loss
loss = criterion(logits, targets)

# Backward pass
optimizer.zero_grad()
loss.backward()

# Clip gradients (Critical for RNNs, mentioned in paper references)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

optimizer.step()

print(f"Step complete. Loss: {loss.item():.4f}")

