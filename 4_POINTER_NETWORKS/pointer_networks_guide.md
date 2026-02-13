# Pointer Networks: Step-by-Step Implementation Guide

## Paper Overview
**Pointer Networks** (Vinyals et al., 2015) solve problems where:
- Output is a sequence of positions/indices from the input
- Output dictionary size = input sequence length (variable!)
- Examples: Convex Hull, TSP, Sorting

**Key Innovation**: Use attention mechanism as a **pointer** to select input elements, not to blend hidden states.

---

## Architecture Components

### 1. Encoder (LSTM)
Processes input sequence P = {P₁, ..., Pₙ} → produces hidden states e₁, ..., eₙ

### 2. Decoder (LSTM) 
At each step i, produces hidden state dᵢ

### 3. Pointer Mechanism (Attention as Pointer)
```
uⁱⱼ = vᵀ tanh(W₁eⱼ + W₂dᵢ)    for j ∈ {1, ..., n}
p(Cᵢ | C₁...Cᵢ₋₁, P) = softmax(uⁱ)
```

**Critical difference from normal attention**: 
- Normal attention: uses softmax weights to blend encoder states
- Pointer Net: uses softmax directly as output distribution over input positions

---

## Implementation Steps

We'll implement using the **Convex Hull** problem as example:
- Input: Set of 2D points
- Output: Sequence of point indices forming the convex hull

### Step 1: Data Generation
### Step 2: Build Encoder
### Step 3: Build Decoder with Pointer Mechanism
### Step 4: Build Complete Model
### Step 5: Training Loop
### Step 6: Inference with Beam Search
### Step 7: Evaluation & Visualization

---

## Mathematical Details

### Forward Pass
1. Encoder: Process all input points
   ```
   e₁, ..., eₙ = LSTM_encoder(P₁, ..., Pₙ)
   ```

2. Decoder: For each output step i
   ```
   dᵢ = LSTM_decoder(P_{Cᵢ₋₁}, dᵢ₋₁)
   uⁱ = vᵀ tanh(W₁E + W₂dᵢ)     # E is matrix of all encoder states
   p(Cᵢ) = softmax(uⁱ)
   ```

### Loss Function
Negative log-likelihood:
```
L = -∑ᵢ log p(Cᵢ* | C₁*, ..., Cᵢ₋₁*, P)
```
where Cᵢ* is the ground truth index at position i

---

## Key Implementation Tricks

1. **Teacher Forcing**: During training, feed ground truth previous output
2. **Masking**: For inference, mask already selected indices (for problems like TSP)
3. **Beam Search**: Use beam search for better solutions at inference
4. **Input Ordering**: Order in which we present inputs matters

---

## Practice Exercises

### Exercise 1: Implement basic encoder-decoder
### Exercise 2: Add attention mechanism
### Exercise 3: Convert attention to pointer mechanism
### Exercise 4: Implement convex hull solver
### Exercise 5: Add beam search
### Exercise 6: Extend to TSP problem

Each exercise builds on the previous one!
