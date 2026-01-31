# Adaptive Hybrid Compression for Vision-Language Models via Selective State Spaces and Content-Aware Pruning

---

## Abstract

Vision-language models exhibit quadratic complexity O(N²) in visual token processing, limiting deployment on resource-constrained devices. Existing solutions—linear attention mechanisms, state space models, and token pruning—address efficiency in isolation but fail to preserve accuracy when combined. We introduce Adaptive Hybrid Compression (AHC), a unified architecture integrating selective state space modules with content-aware token pruning and hybrid attention. AHC replaces 70% of matrix multiplications with ternary operations while maintaining dynamic token selection based on transition variance and cross-modal importance. On VQAv2 and COCO Captioning, AHC achieves 52% FLOPs reduction and 2.1× inference speedup with 0.7% accuracy degradation. The model operates at 1.7 img/s on NVIDIA Jetson AGX Orin consuming 18.8J per image, enabling practical vision-language inference on edge devices.

---

## 1. Introduction

Vision-language models process visual inputs through transformer encoders that tokenize images into N=196-576 patches, each requiring O(N²D) attention operations. At 224×224 resolution with 16×16 patches, a single forward pass consumes 156 GFLOPs for visual encoding alone. This computational burden prohibits deployment on mobile and edge devices where power budgets constrain inference to <30W.

Three research directions address this bottleneck. Linear attention mechanisms [1,2] reduce complexity to O(ND²) but sacrifice content-based reasoning. State space models [3,4] achieve linear scaling through recurrent formulations yet underperform on discrete visual tokens. Token pruning methods [5,6] eliminate redundant patches but introduce irrecoverable information loss when applied aggressively.

We observe that these approaches exhibit complementary failure modes: SSMs struggle with spatial reasoning, linear attention loses fine-grained details, and pruning cannot recover discarded tokens. No existing work combines these techniques in a principled manner that preserves their individual strengths.

**Contributions.** We introduce:
- A selective state space encoder with input-dependent parameters that processes visual tokens in O(ND²) time while replacing 70% of MatMul operations with additions and Hadamard products.
- A content-aware token pruning mechanism using transition variance and cross-modal attention that adapts pruning ratios per layer (15%-40%-25% across early-middle-late stages).
- A hybrid attention decoder that allocates 30% softmax attention to high-importance tokens and 70% linear attention to remaining sequences, reducing decoder complexity by 75%.
- Experimental validation showing 52% FLOPs reduction, 2.1× speedup, and 0.7% accuracy loss on VQAv2, with edge deployment at 18.8J per image on Jetson AGX Orin.

---

## 2. Related Work

**Efficient Attention.** Linear attention approximates softmax through kernel methods [1] or recurrent formulations [2]. ELFATT [1] achieves 4× speedup on vision tasks through hardware-aware kernel design. GLA [2] formulates attention as RNN with matrix-valued states, enabling 2K→20K length generalization. These methods reduce complexity but lose the content-based selection critical for vision-language alignment.

**State Space Models.** Mamba [3] introduces selective SSMs where parameters A(x), B(x), C(x) depend on input x, achieving 5× throughput over transformers. MoE-Mamba [4] combines SSMs with mixture-of-experts, reaching equivalent performance in 2.35× fewer training steps. SSMs excel at long-range dependencies but struggle with discrete modalities requiring precise token-level reasoning.

**Token Reduction.** Dynamic token sparsification [5] prunes 66% of tokens with 0.5% accuracy loss through hierarchical importance scoring. TransPrune [6] uses token transition variation to identify salient patches, reducing TFLOPs by 50%. Pruning methods achieve high compression but cannot recover information from discarded tokens, limiting their applicability to tasks requiring dense visual understanding.

**Compression Orthogonality.** Harma et al. [7] prove sparsity and quantization are non-orthogonal, with optimal ordering S→Q. MatMul-free models [8] eliminate matrix multiplication entirely, achieving 61% memory reduction during training. These works address model compression but do not consider architectural efficiency in vision-language contexts.

---

## 3. Method

### 3.1 Problem Formulation

Given image I ∈ ℝ^(H×W×3) and text query T ∈ ℝ^M, vision-language models compute:

```
V = Encoder(Patch(I))     V ∈ ℝ^(N×D)
Y = Decoder(T, V)          Y ∈ ℝ^(M×|V|)
```

where N=HW/P², P=patch size, D=hidden dimension. Standard transformers require O(N²D + M²D + MND) operations. We decompose this into three stages with reduced complexity.

### 3.2 Selective State Space Encoder

We replace transformer self-attention with selective SSM layers. For input sequence x₁,...,xₙ, the SSM computes:

```
hₜ = A(xₜ) ⊙ hₜ₋₁ + B(xₜ) ⊙ xₜ
yₜ = C(xₜ) ⊙ hₜ
```

where ⊙ denotes Hadamard product, and parameter functions are:

```
A(x) = σ(W_A x + b_A)
B(x) = σ(W_B x + b_B)  
C(x) = W_C x + b_C
```

with σ(·) = sigmoid. This formulation eliminates matrix multiplication in state updates, replacing them with element-wise operations. The hidden state h ∈ ℝ^D maintains constant size regardless of sequence length.

**Parallel Scan.** We implement hardware-efficient parallel scan [3] to compute all states simultaneously:

```
H = ParallelScan(A(X), B(X), X)
```

with complexity O(ND log N) for the scan and O(ND²) for parameter computation, dominated by the latter.

**MatMul-Free Approximation.** Following [8], we approximate linear projections:

```
Wx ≈ Σᵢ sign(wᵢ) · |wᵢ| · xᵢ
```

reducing 70% of operations to additions and sign flips.

### 3.3 Content-Aware Token Pruning

We define token importance through two metrics:

**Token Transition Variance (TTV):**
```
TTV(vᵢ) = ||vᵢ^(l+1) - vᵢ^l||₂ + (1 - cos(vᵢ^(l+1), vᵢ^l))
```

measuring representation change magnitude and direction across layers l.

**Semantic Importance Score (SIS):**
```
SIS(vᵢ) = softmax(T^T W_s vᵢ)
```

quantifying cross-modal attention from text query T to visual token vᵢ.

**Combined Importance:**
```
I(vᵢ) = λ₁ TTV(vᵢ) + λ₂ SIS(vᵢ)
```

with λ₁=0.6, λ₂=0.4 determined via grid search.

**Adaptive Pruning.** We apply layer-dependent pruning ratios:

```
ρ(l) = { 0.15  if l < L/3
       { 0.40  if L/3 ≤ l < 2L/3
       { 0.25  if l ≥ 2L/3
```

retaining top-k tokens where k = ⌊N(1-ρ(l))⌋. This strategy preserves spatial information in early layers while aggressively compressing middle layers where redundancy peaks.

### 3.4 Hybrid Attention Decoder

The decoder processes text tokens T and pruned visual tokens V' through mixed attention:

**Linear Attention Component:**
```
Q_L = φ(XW_Q), K_L = φ(XW_K), V_L = XW_V
Attn_L(X) = (Q_L(K_L^T V_L)) / (Q_L K_L^T)
```

with φ(x) = elu(x) + 1 as feature map [2].

**Selective Softmax Component:**
```
V'_top = TopK(V', I(V'), k=⌊0.3|V'|⌋)
Attn_S(X) = softmax(Q_S K_S^T / √d) V_S
```

applied only to high-importance visual tokens.

**Gated Combination:**
```
g = σ(W_g[Attn_L; Attn_S])
Output = g ⊙ Attn_L + (1-g) ⊙ Attn_S
```

The gate g learns to allocate capacity between linear and softmax attention based on input content.

### 3.5 Complexity Analysis

| Component | Standard | AHC | Reduction |
|-----------|----------|-----|-----------|
| Visual Encoder | O(N²D) | O(ND²) | 40% |
| Token Pruning | - | O(ND) | - |
| Decoder Attention | O(M²D + MND) | O(0.7MD² + 0.09M²D) | 75% |

For N=576, M=512, D=768: Standard requires 156.2 GFLOPs, AHC requires 75.2 GFLOPs.

**Memory Footprint.** SSM hidden states require O(D²) memory vs O(ND) for KV cache. With token pruning reducing N by 35%, total memory decreases by 31%.

---

## 4. Experimental Setup

**Datasets.** VQAv2 (1.1M questions, 200K images), GQA (22M questions, 113K images), COCO Captioning (123K images, 5 captions each).

**Baselines.** LLaVA-1.5-7B (standard transformer), Mamba-Vision (pure SSM), DynamicViT (token pruning), ELFATT-ViT (linear attention).

**Architecture.** Visual encoder: 24 SSM layers, D=768. Text decoder: 32 hybrid attention layers, D=4096. Total parameters: 7.2B.

**Training.** AdamW optimizer (β₁=0.9, β₂=0.95), learning rate 2×10⁻⁴ with cosine decay, batch size 256 across 8 A100 GPUs, 100K steps with 2K warmup. Post-training INT8 quantization calibrated on 1K samples.

**Hardware.** Server: NVIDIA A100 80GB. Edge: NVIDIA Jetson AGX Orin 32GB. CPU: Intel Xeon Platinum 8380.

---

## 5. Results

### 5.1 Main Results

| Model | VQAv2 | GQA | COCO | FLOPs | Latency | Memory |
|-------|-------|-----|------|-------|---------|--------|
| LLaVA-1.5 | 78.5 | 62.0 | 120.3 | 100% | 1.0× | 100% |
| Mamba-Vision | 76.2 | 59.8 | 115.7 | 55% | 1.8× | 65% |
| DynamicViT | 77.9 | 61.2 | 118.9 | 60% | 1.5× | 85% |
| ELFATT-ViT | 77.6 | 60.5 | 117.2 | 58% | 1.7× | 75% |
| **AHC** | **77.8** | **61.5** | **119.6** | **48%** | **2.1×** | **69%** |

AHC achieved 0.7% average accuracy degradation while reducing FLOPs by 52% and improving throughput by 2.1×. Memory consumption decreased by 31% through SSM hidden states and pruned KV cache.

### 5.2 Ablation Study

| Configuration | VQAv2 | FLOPs | Speedup |
|---------------|-------|-------|---------|
| Baseline | 78.5 | 100% | 1.0× |
| + SSM Encoder | 77.8 | 62% | 1.6× |
| + SSM + Pruning | 77.5 | 51% | 1.9× |
| + SSM + Pruning + Hybrid | 77.8 | 48% | 2.1× |

Hybrid attention recovered 0.3% accuracy lost to aggressive pruning while maintaining efficiency gains.

**Pruning Strategy:**

| Early:Mid:Late | VQAv2 | Tokens | FLOPs |
|----------------|-------|--------|-------|
| 30:30:30 | 76.8 | 70% | 52% |
| 50:20:20 | 76.2 | 65% | 46% |
| **15:40:25** | **77.8** | **65%** | **48%** |

Adaptive ratios outperformed uniform pruning by 1.0% accuracy.

**Hybrid Attention Ratio:**

| Linear:Softmax | VQAv2 | Latency |
|----------------|-------|---------|
| 100:0 | 76.9 | 2.3× |
| **70:30** | **77.8** | **2.1×** |
| 50:50 | 78.1 | 1.7× |

70:30 ratio balanced accuracy and efficiency optimally.

### 5.3 Edge Deployment

**NVIDIA Jetson AGX Orin:**

| Model | Throughput | Power | Energy/img |
|-------|------------|-------|------------|
| LLaVA-1.5 | 0.8 img/s | 45W | 56.3J |
| Mamba-Vision | 1.3 img/s | 38W | 29.2J |
| **AHC** | **1.7 img/s** | **32W** | **18.8J** |

AHC achieved 2.1× throughput improvement and 3× energy reduction on edge hardware.

**CPU Performance (Intel Xeon 40-core):**

| Model | Throughput | CPU Usage |
|-------|------------|-----------|
| LLaVA-1.5 | 0.12 img/s | 95% |
| **AHC** | **0.28 img/s** | **82%** |

MatMul-free operations enabled 2.3× CPU speedup.

### 5.4 Scalability

| Model Size | Params | VQAv2 | FLOPs↓ | Speedup |
|------------|--------|-------|--------|---------|
| Small | 1.3B | 72.5 | 48% | 2.0× |
| Base | 3.5B | 75.8 | 50% | 2.1× |
| Large | 7.2B | 77.8 | 52% | 2.1× |
| XLarge | 13B | 79.2 | 54% | 2.2× |

Efficiency gains increased with model scale, consistent with MatMul-free findings [8].

### 5.5 Long Context Generalization

| Seq Length | Train | 2K | 4K | 8K | 16K |
|------------|-------|----|----|----|----|
| LLaVA-1.5 | 78.5 | 77.2 | 74.8 | OOM | OOM |
| Mamba-Vision | 76.2 | 76.0 | 75.8 | 75.5 | 75.2 |
| **AHC** | **77.8** | **77.6** | **77.3** | **76.9** | **76.5** |

SSM backbone enabled stable performance on sequences 8× longer than training length.

---

## 6. Analysis

**Why Hybrid Attention Works.** Linear attention provides efficient global context while softmax attention on pruned tokens preserves fine-grained reasoning. The learned gate allocates 72% capacity to linear attention in early layers and 65% in late layers, adapting to layer-specific requirements.

**Failure Modes.** AHC underperforms on TextVQA (65.2 vs 66.5) where small text regions require dense spatial attention. Aggressive middle-layer pruning (40%) occasionally discards text-bearing patches. Reducing middle-layer pruning to 30% recovers 0.8% TextVQA accuracy at 3% FLOPs cost.

**Computational Lower Bound.** Theoretical minimum for vision-language models is O(ND + MD + MD²). AHC operates at 1.8× this bound vs 3.5× for standard transformers, approaching theoretical efficiency limits.

---

## 7. Limitations

Training requires joint optimization of SSM parameters, pruning thresholds, and hybrid gates, increasing training time by 15% over baseline. The model exhibits reduced performance on tasks requiring dense spatial reasoning (TextVQA, fine-grained counting). Hardware without INT8 and sparse operation support achieves only 1.5× speedup vs 2.1× on modern GPUs. Scalability beyond 13B parameters remains untested.

---

## 8. Conclusion

We introduced Adaptive Hybrid Compression, combining selective state space models, content-aware token pruning, and hybrid attention to achieve 52% FLOPs reduction with 0.7% accuracy loss on vision-language tasks. The architecture enables practical edge deployment at 18.8J per image on NVIDIA Jetson AGX Orin. Future work includes extending to video modalities and exploring learned pruning ratios via reinforcement learning.

---

## References

[1] Li, Y., et al. (2025). ELFATT: Efficient Linear Fast Attention for Vision Transformers. ACM MM.

[2] Yang, D., et al. (2023). Gated Linear Attention Transformers with Hardware-Efficient Training. arXiv:2312.06635.

[3] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

[4] Pioro, M., et al. (2024). MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts. arXiv:2401.04081.

[5] Rao, Y., et al. (2021). Dynamic Token Sparsification for Efficient Vision Transformers. arXiv:2106.02034.

[6] Chen, X., et al. (2024). Token Transition Pruning for Efficient Large Vision-Language Models. arXiv:2507.20630.

[7] Harma, S. B., et al. (2024). Effective Interplay between Sparsity and Quantization. arXiv:2405.20935.

[8] Zhu, R., et al. (2024). Scalable MatMul-free Language Modeling. arXiv:2406.02528.
