# Project Structure

## 📁 Directory Layout

```
adaptive-hybrid-compression/
│
├── 📄 README.md                          # Main documentation (English)
├── 📄 SUMMARY_AR.md                      # Arabic summary
├── 📄 Hybrid_Efficient_Architecture_Paper.md  # Full research paper
├── 📄 LICENSE                            # MIT License
├── 📄 CONTRIBUTING.md                    # Contribution guidelines
├── 📄 CITATION.cff                       # Citation metadata
├── 📄 requirements.txt                   # Python dependencies
├── 📄 requirements-dev.txt               # Development dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 📂 models/                            # Model implementations
│   ├── __init__.py
│   ├── ssm_encoder.py                   # Selective SSM Encoder (S³E)
│   ├── token_pruner.py                  # Content-Aware Token Pruner (CATP)
│   ├── hybrid_attention.py              # Hybrid Attention Decoder (HAD)
│   ├── ahc_model.py                     # Full AHC model
│   └── utils.py                         # Model utilities
│
├── 📂 kernels/                           # Custom CUDA kernels
│   ├── __init__.py
│   ├── parallel_scan.cu                 # Parallel scan for SSM
│   ├── matmul_free.cu                   # MatMul-free operations
│   ├── sparse_attention.cu              # Sparse attention kernel
│   └── setup.py                         # Kernel compilation
│
├── 📂 training/                          # Training scripts
│   ├── __init__.py
│   ├── train.py                         # Main training script
│   ├── pretrain_ssm.py                  # SSM pre-training
│   ├── data_loader.py                   # Data loading utilities
│   ├── optimizer.py                     # Custom optimizer
│   └── trainer.py                       # Training loop
│
├── 📂 evaluation/                        # Evaluation scripts
│   ├── __init__.py
│   ├── eval_vqa.py                      # VQA evaluation
│   ├── eval_caption.py                  # Captioning evaluation
│   ├── eval_textvqa.py                  # TextVQA evaluation
│   ├── benchmark.py                     # Efficiency benchmarks
│   └── metrics.py                       # Evaluation metrics
│
├── 📂 deployment/                        # Deployment tools
│   ├── __init__.py
│   ├── export_onnx.py                   # ONNX export
│   ├── tensorrt_convert.py              # TensorRT conversion
│   ├── quantize.py                      # INT8 quantization
│   └── optimize_edge.py                 # Edge device optimization
│
├── 📂 configs/                           # Configuration files
│   ├── ahc_small.yaml                   # Small model (1.3B)
│   ├── ahc_base.yaml                    # Base model (3.5B)
│   ├── ahc_large.yaml                   # Large model (7.2B)
│   ├── ahc_xlarge.yaml                  # XLarge model (13B)
│   ├── training.yaml                    # Training config
│   └── deployment.yaml                  # Deployment config
│
├── 📂 tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_ssm_encoder.py
│   ├── test_token_pruner.py
│   ├── test_hybrid_attention.py
│   ├── test_ahc_model.py
│   └── benchmark_*.py                   # Benchmark tests
│
├── 📂 notebooks/                         # Jupyter notebooks
│   ├── 01_quick_start.ipynb
│   ├── 02_token_pruning_analysis.ipynb
│   ├── 03_attention_visualization.ipynb
│   └── 04_efficiency_profiling.ipynb
│
├── 📂 docs/                              # Documentation
│   ├── index.md
│   ├── installation.md
│   ├── quickstart.md
│   ├── architecture.md
│   ├── training.md
│   ├── deployment.md
│   └── api/                             # API documentation
│
├── 📂 scripts/                           # Utility scripts
│   ├── download_data.sh                 # Download datasets
│   ├── setup_env.sh                     # Environment setup
│   ├── run_benchmarks.sh                # Run all benchmarks
│   └── profile_model.py                 # Model profiling
│
└── 📂 assets/                            # Assets
    ├── figures/                         # Paper figures
    ├── results/                         # Experimental results
    └── checkpoints/                     # Pre-trained checkpoints
```

## 🔑 Key Files

### Core Implementation

| File | Description | Status |
|------|-------------|--------|
| `models/ssm_encoder.py` | Selective SSM with O(N·D²) complexity | 🔄 To implement |
| `models/token_pruner.py` | Adaptive token pruning (TTV + SIS) | 🔄 To implement |
| `models/hybrid_attention.py` | 70% linear + 30% softmax attention | 🔄 To implement |
| `models/ahc_model.py` | Full AHC architecture | 🔄 To implement |

### CUDA Kernels

| File | Description | Status |
|------|-------------|--------|
| `kernels/parallel_scan.cu` | Hardware-aware parallel scan | 🔄 To implement |
| `kernels/matmul_free.cu` | MatMul-free approximations | 🔄 To implement |
| `kernels/sparse_attention.cu` | Sparse attention operations | 🔄 To implement |

### Training & Evaluation

| File | Description | Status |
|------|-------------|--------|
| `training/train.py` | Main training pipeline | 🔄 To implement |
| `evaluation/eval_vqa.py` | VQA evaluation | 🔄 To implement |
| `evaluation/benchmark.py` | FLOPs/latency/memory benchmarks | 🔄 To implement |

### Documentation

| File | Description | Status |
|------|-------------|--------|
| `README.md` | Main documentation | ✅ Complete |
| `SUMMARY_AR.md` | Arabic summary | ✅ Complete |
| `Hybrid_Efficient_Architecture_Paper.md` | Full paper | ✅ Complete |
| `CONTRIBUTING.md` | Contribution guide | ✅ Complete |

## 🚀 Implementation Roadmap

### Phase 1: Core Components (Weeks 1-4)
- [ ] Implement SSM encoder
- [ ] Implement token pruner
- [ ] Implement hybrid attention
- [ ] Unit tests for each component

### Phase 2: Integration (Weeks 5-6)
- [ ] Integrate components into full model
- [ ] End-to-end forward pass
- [ ] Memory profiling
- [ ] Initial benchmarks

### Phase 3: Training (Weeks 7-10)
- [ ] Pre-train SSM encoder on ImageNet
- [ ] Fine-tune on VQA datasets
- [ ] Hyperparameter tuning
- [ ] Ablation studies

### Phase 4: Optimization (Weeks 11-12)
- [ ] CUDA kernel optimization
- [ ] Quantization (INT8)
- [ ] TensorRT deployment
- [ ] Edge device testing

### Phase 5: Evaluation (Weeks 13-14)
- [ ] Full benchmark suite
- [ ] Comparison with baselines
- [ ] Failure case analysis
- [ ] Documentation

## 📊 Expected Deliverables

1. **Code:**
   - Fully functional AHC implementation
   - Custom CUDA kernels
   - Training and evaluation scripts

2. **Models:**
   - Pre-trained checkpoints (1.3B, 3.5B, 7.2B, 13B)
   - Quantized models (INT8)
   - ONNX/TensorRT exports

3. **Documentation:**
   - API documentation
   - Tutorial notebooks
   - Deployment guides

4. **Results:**
   - Benchmark results
   - Ablation studies
   - Comparison tables

## 🔗 Dependencies

### Core
- PyTorch ≥2.0
- CUDA ≥11.8
- FlashAttention-2

### Optional
- TensorRT (deployment)
- Intel Extension for PyTorch (CPU optimization)
- Weights & Biases (experiment tracking)

## 📝 Notes

- All code should follow PEP 8 style guidelines
- Use type hints for better code clarity
- Write comprehensive docstrings
- Include unit tests for new features
- Profile performance-critical code

---

**Last Updated:** February 2026  
**Status:** Research paper complete, implementation in progress
