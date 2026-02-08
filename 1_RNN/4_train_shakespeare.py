"""
Train RNN on Shakespeare - recreating the famous example from the article!

This shows how to:
1. Download/prepare Shakespeare text
2. Train an RNN on it
3. Generate Shakespeare-like text
4. Analyze what the model learned
"""

import numpy as np
import requests
import matplotlib.pyplot as plt
from collections import Counter


def download_shakespeare():
    """Download Shakespeare's complete works"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    print("Downloading Shakespeare text...")
    try:
        response = requests.get(url)
        text = response.text
        
        # Save locally
        with open('/home/claude/shakespeare.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Downloaded {len(text)} characters")
        return text
    except Exception as e:
        print(f"Could not download: {e}")
        print("Using sample Shakespeare instead...")
        return get_sample_shakespeare()


def get_sample_shakespeare():
    """Sample Shakespeare text if download fails"""
    return """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have had
inkling this fortnight what we intend to do, which now
we'll show 'em in deeds. They say poor suitors have
strong breaths: they shall know we have strong arms too.

ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.
It is my lady, O, it is my love!
O, that she knew she were!

HAMLET:
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
""" * 20  # Repeat to have more data


class ShakespeareRNN:
    """Simple RNN for Shakespeare generation"""
    
    def __init__(self, data, hidden_size=256):
        self.data = data
        self.chars = sorted(list(set(data)))
        self.data_size = len(data)
        self.vocab_size = len(self.chars)
        self.hidden_size = hidden_size
        
        # Character mappings
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Model parameters
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))
        
        # Adagrad memory
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
        print(f"\nShakespeare RNN initialized:")
        print(f"  Data size: {self.data_size:,} characters")
        print(f"  Vocabulary: {self.vocab_size} unique characters")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Parameters: {self.count_parameters():,}\n")
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return (self.Wxh.size + self.Whh.size + self.Why.size + 
                self.bh.size + self.by.size)
    
    def analyze_data(self):
        """Analyze the Shakespeare dataset"""
        char_freq = Counter(self.data)
        
        print("="*60)
        print("DATA ANALYSIS")
        print("="*60)
        print(f"Total characters: {len(self.data):,}")
        print(f"Unique characters: {len(self.chars)}")
        print(f"\nMost common characters:")
        for char, count in char_freq.most_common(10):
            if char == '\n':
                char_display = '\\n'
            elif char == ' ':
                char_display = 'SPACE'
            else:
                char_display = char
            print(f"  '{char_display}': {count:,} ({100*count/len(self.data):.1f}%)")
        
        # Count lines and words
        lines = self.data.split('\n')
        words = self.data.split()
        print(f"\nApproximate statistics:")
        print(f"  Lines: {len(lines):,}")
        print(f"  Words: {len(words):,}")
        print(f"  Avg word length: {len(self.data)/len(words):.1f} chars")
        
    def train(self, iterations=5000, seq_length=100, learning_rate=0.1):
        """Train the model"""
        n, p = 0, 0
        hprev = np.zeros((self.hidden_size, 1))
        losses = []
        smooth_loss = -np.log(1.0/self.vocab_size) * seq_length
        
        print("Training Shakespeare RNN...")
        print("="*60)
        
        for n in range(iterations):
            # Reset at end of data
            if p + seq_length + 1 >= len(self.data):
                hprev = np.zeros((self.hidden_size, 1))
                p = 0
            
            # Get batch
            inputs = [self.char_to_ix[ch] for ch in self.data[p:p+seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+seq_length+1]]
            
            # Forward pass
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.loss_fun(
                inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            losses.append(smooth_loss)
            
            # Sample periodically
            if n % 500 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print(f"\nIteration {n}, loss: {smooth_loss:.4f}")
                print(f"Sample:\n{txt}\n")
                print("-"*60)
            
            # Update parameters
            for param, dparam, mem in zip(
                [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                [dWxh, dWhh, dWhy, dbh, dby],
                [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
            ):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
            
            p += seq_length
        
        return losses
    
    def loss_fun(self, inputs, targets, hprev):
        """Forward and backward pass"""
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        # Forward
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + 
                           np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])
        
        # Backward
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
    
    def sample(self, h, seed_ix, n, temperature=1.0):
        """Generate text"""
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        
        return ixes
    
    def generate_samples(self, num_samples=3, length=500):
        """Generate multiple samples"""
        print("\n" + "="*60)
        print("GENERATED SHAKESPEARE SAMPLES")
        print("="*60)
        
        h = np.zeros((self.hidden_size, 1))
        for i in range(num_samples):
            seed = np.random.randint(0, self.vocab_size)
            sample_ix = self.sample(h, seed, length, temperature=0.8)
            txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
            print(f"\nSample {i+1}:")
            print(txt)
            print("\n" + "-"*60)


def main():
    """Main training pipeline"""
    # Get data
    try:
        with open('/home/claude/shakespeare.txt', 'r', encoding='utf-8') as f:
            data = f.read()
    except:
        data = download_shakespeare()
    
    # Create and train model
    rnn = ShakespeareRNN(data, hidden_size=128)
    rnn.analyze_data()
    
    # Train
    losses = rnn.train(iterations=2000, seq_length=100, learning_rate=0.1)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss - Shakespeare RNN')
    plt.grid(True)
    plt.savefig('/home/claude/shakespeare_loss.png', dpi=100, bbox_inches='tight')
    print("\nLoss plot saved!")
    
    # Generate samples
    rnn.generate_samples(num_samples=3, length=400)
    
    # Show temperature effect
    print("\n" + "="*60)
    print("TEMPERATURE EFFECTS")
    print("="*60)
    
    h = np.zeros((rnn.hidden_size, 1))
    for temp in [0.5, 1.0, 1.5]:
        sample_ix = rnn.sample(h, rnn.char_to_ix['\n'], 200, temperature=temp)
        txt = ''.join(rnn.ix_to_char[ix] for ix in sample_ix)
        print(f"\nTemperature {temp}:")
        print(txt[:150])


if __name__ == "__main__":
    main()
