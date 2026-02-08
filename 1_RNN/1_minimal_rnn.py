"""
Minimal character-level Vanilla RNN model
Based on Andrej Karpathy's minimal implementation

This is the SIMPLEST possible RNN - great for understanding the core concepts!
We'll train it to predict the next character in a sequence.
"""

import numpy as np

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
hidden_size = 100      # Size of hidden layer of neurons
seq_length = 25        # Number of steps to unroll the RNN for backprop
learning_rate = 1e-1   # Learning rate for gradient descent

# ============================================================================
# DATA PREPARATION
# ============================================================================
# Read input data - you can use any text file!
data = open('input.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))               # unique characters
data_size, vocab_size = len(data), len(chars)
print(f'Data has {data_size} characters, {vocab_size} unique.')

# Create mappings between characters and indices
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# ============================================================================
# MODEL PARAMETERS (weights and biases)
# ============================================================================
# These are the parameters that will be learned during training!
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01   # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01   # hidden to output
bh = np.zeros((hidden_size, 1))                         # hidden bias
by = np.zeros((vocab_size, 1))                          # output bias


def lossFun(inputs, targets, hprev):
    """
    Forward pass and loss computation

    Args:
        inputs: list of integers (character indices)
        targets: list of integers (what character should come next)
        hprev: hidden state from previous sequence (Hx1 array)

    Returns:
        loss: cross-entropy loss
        dWxh, dWhh, dWhy, dbh, dby: gradients for parameters
        hprev: last hidden state (to continue sequence)
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # FORWARD PASS
    for t in range(len(inputs)):
        # Encode input character as one-hot vector
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        # Hidden state: h = tanh(Wxh * x + Whh * h_prev + bh)
        # This is the CORE RNN equation!
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)

        # Output (unnormalized log probabilities): y = Why * h + by
        ys[t] = np.dot(Why, hs[t]) + by

        # Probabilities for next character (using softmax)
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        # Cross-entropy loss: -log(probability of correct character)
        loss += -np.log(ps[t][targets[t], 0])

    # BACKWARD PASS (backpropagation through time)
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        # Backprop through output layer
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # derivative of softmax with cross-entropy

        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        # Backprop into hidden layer
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh

        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

    # Clip gradients to prevent exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


def sample(h, seed_ix, n):
    """
    Sample a sequence of characters from the model

    Args:
        h: initial hidden state
        seed_ix: seed character index
        n: number of characters to sample

    Returns:
        ixes: list of sampled character indices
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        # Forward pass (same as in training)
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))

        # Sample next character from probability distribution
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        # Use sampled character as input for next step
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

    return ixes


# ============================================================================
# TRAINING LOOP
# ============================================================================
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size) * seq_length  # loss at iteration 0

while True:
    # Prepare inputs (we're sweeping from left to right through data)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go back to start of data

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # Sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    # Forward pass
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 100 == 0:
        print(f'iter {n}, loss: {smooth_loss:.4f}')

    # Parameter update with Adagrad
    for param, dparam, mem in zip(
        [Wxh, Whh, Why, bh, by],
        [dWxh, dWhh, dWhy, dbh, dby],
        [mWxh, mWhh, mWhy, mbh, mby]
    ):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1
