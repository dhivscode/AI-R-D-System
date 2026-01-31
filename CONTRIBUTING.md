# Contributing to Adaptive Hybrid Compression (AHC)

Thank you for your interest in contributing to AHC! This document provides guidelines for contributing to the project.

## 🎯 Areas for Contribution

### 1. Implementation
- [ ] SSM encoder with custom CUDA kernels
- [ ] Token pruning module
- [ ] Hybrid attention mechanism
- [ ] Training pipeline
- [ ] Evaluation scripts

### 2. Optimization
- [ ] Hardware-specific optimizations (GPU/CPU/Edge)
- [ ] Quantization strategies
- [ ] Sparse kernel implementations
- [ ] Memory optimization

### 3. Experiments
- [ ] Benchmark on additional datasets
- [ ] Ablation studies
- [ ] Hyperparameter tuning
- [ ] Failure case analysis

### 4. Documentation
- [ ] Code documentation
- [ ] Tutorial notebooks
- [ ] Deployment guides
- [ ] Performance profiling

## 🔧 Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/adaptive-hybrid-compression.git
cd adaptive-hybrid-compression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## 📝 Code Style

- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Write unit tests for new features

### Example:
```python
def compute_importance(
    tokens: torch.Tensor,
    text_features: torch.Tensor,
    lambda1: float = 0.6,
    lambda2: float = 0.4
) -> torch.Tensor:
    """
    Compute token importance scores.
    
    Args:
        tokens: Visual tokens [B, N, D]
        text_features: Text features [B, M, D]
        lambda1: Weight for TTV
        lambda2: Weight for SIS
        
    Returns:
        Importance scores [B, N]
    """
    ttv = compute_ttv(tokens)
    sis = compute_sis(tokens, text_features)
    return lambda1 * ttv + lambda2 * sis
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_token_pruner.py

# Run with coverage
pytest --cov=models tests/
```

## 📊 Benchmarking

When adding new features, include benchmarks:

```python
# tests/benchmark_pruning.py
import time
import torch

def benchmark_pruning():
    model = TokenPruner()
    tokens = torch.randn(1, 576, 768).cuda()
    
    # Warmup
    for _ in range(10):
        _ = model(tokens)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = model(tokens)
    elapsed = time.time() - start
    
    print(f"Average time: {elapsed/100*1000:.2f}ms")
```

## 🔀 Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests:**
   ```bash
   pytest tests/
   python -m flake8 models/
   python -m mypy models/
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add token pruning optimization"
   ```
   
   Use conventional commits:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `perf:` Performance improvement
   - `test:` Tests
   - `refactor:` Code refactoring

5. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## 📋 PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Benchmarks included (if applicable)
- [ ] PR description clearly explains changes

## 🐛 Reporting Bugs

When reporting bugs, include:
- Python version
- PyTorch version
- CUDA version (if using GPU)
- Hardware specifications
- Minimal reproducible example
- Error messages and stack traces

## 💡 Suggesting Enhancements

For feature requests:
- Clearly describe the feature
- Explain the use case
- Provide examples if possible
- Discuss potential implementation approaches

## 📧 Contact

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and general discussion

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AHC! 🚀
