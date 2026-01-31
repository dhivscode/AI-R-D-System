# Quick Start Guide

Get started with Adaptive Hybrid Compression (AHC) in 5 minutes.

## 📦 Installation

### Option 1: From Source (Recommended)

```bash
# Clone repository
git clone https://github.com/username/adaptive-hybrid-compression.git
cd adaptive-hybrid-compression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install custom CUDA kernels (optional, for best performance)
cd kernels
python setup.py install
cd ..
```

### Option 2: Using pip (Coming Soon)

```bash
pip install adaptive-hybrid-compression
```

## 🚀 Basic Usage

### 1. Load Pre-trained Model

```python
from models import AHCModel

# Load model (7.2B parameters)
model = AHCModel.from_pretrained("ahc-7b")
model.eval()
model.cuda()  # Move to GPU

print(f"Model loaded: {model.num_parameters() / 1e9:.1f}B parameters")
```

### 2. Run Inference

```python
from PIL import Image
import torch

# Load image
image = Image.open("example.jpg")

# Prepare input
question = "What is in this image?"
inputs = model.prepare_inputs(image, question)

# Generate answer
with torch.no_grad():
    output = model.generate(**inputs, max_length=50)
    answer = model.decode(output)

print(f"Q: {question}")
print(f"A: {answer}")
```

### 3. Batch Inference

```python
# Multiple images and questions
images = [Image.open(f"img{i}.jpg") for i in range(4)]
questions = [
    "What is in this image?",
    "How many people are there?",
    "What color is the car?",
    "What is the weather like?"
]

# Batch processing
inputs = model.prepare_batch(images, questions)
outputs = model.generate(**inputs, max_length=50)
answers = model.decode_batch(outputs)

for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")
```

## ⚙️ Configuration

### Adjust Pruning Ratio

```python
# More aggressive pruning (faster, slightly lower accuracy)
model.config.pruning_ratios = [0.20, 0.50, 0.30]  # early, middle, late

# Conservative pruning (slower, higher accuracy)
model.config.pruning_ratios = [0.10, 0.30, 0.20]
```

### Adjust Hybrid Attention Ratio

```python
# More linear attention (faster)
model.config.linear_attention_ratio = 0.80  # 80% linear, 20% softmax

# More softmax attention (higher accuracy)
model.config.linear_attention_ratio = 0.60  # 60% linear, 40% softmax
```

### Enable/Disable Quantization

```python
# INT8 quantization (faster, lower memory)
model.quantize(dtype=torch.int8)

# FP16 (balanced)
model.half()

# FP32 (highest accuracy)
model.float()
```

## 📊 Benchmarking

### Measure Inference Speed

```python
import time

# Warmup
for _ in range(10):
    _ = model.generate(**inputs)

# Benchmark
start = time.time()
for _ in range(100):
    _ = model.generate(**inputs)
elapsed = time.time() - start

print(f"Average latency: {elapsed/100*1000:.2f}ms")
print(f"Throughput: {100/elapsed:.2f} samples/sec")
```

### Measure Memory Usage

```python
import torch

torch.cuda.reset_peak_memory_stats()

# Run inference
output = model.generate(**inputs)

memory_used = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak memory: {memory_used:.2f} GB")
```

### Profile FLOPs

```python
from fvcore.nn import FlopCountAnalysis

# Count FLOPs
flops = FlopCountAnalysis(model, inputs)
print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
```

## 🎯 Common Use Cases

### 1. Visual Question Answering

```python
image = Image.open("street.jpg")
question = "How many cars are in the image?"
answer = model.vqa(image, question)
print(answer)  # "3"
```

### 2. Image Captioning

```python
image = Image.open("sunset.jpg")
caption = model.caption(image)
print(caption)  # "A beautiful sunset over the ocean"
```

### 3. Visual Reasoning

```python
image = Image.open("chart.jpg")
question = "What is the trend shown in the graph?"
answer = model.reason(image, question)
print(answer)  # "The values are increasing over time"
```

## 🔧 Troubleshooting

### Out of Memory

```python
# Reduce batch size
model.config.batch_size = 1

# Enable gradient checkpointing
model.config.gradient_checkpointing = True

# Use INT8 quantization
model.quantize(dtype=torch.int8)
```

### Slow Inference

```python
# Increase pruning ratio
model.config.pruning_ratios = [0.25, 0.55, 0.35]

# Use more linear attention
model.config.linear_attention_ratio = 0.85

# Enable CUDA kernels
model.config.use_custom_kernels = True
```

### Low Accuracy

```python
# Reduce pruning ratio
model.config.pruning_ratios = [0.10, 0.30, 0.20]

# Use more softmax attention
model.config.linear_attention_ratio = 0.60

# Disable quantization
model.float()
```

## 📚 Next Steps

- **Training:** See [training/README.md](training/README.md)
- **Deployment:** See [deployment/README.md](deployment/README.md)
- **API Reference:** See [docs/api/](docs/api/)
- **Examples:** See [notebooks/](notebooks/)

## 💡 Tips

1. **Start with default settings** and adjust based on your needs
2. **Profile your workload** to identify bottlenecks
3. **Use INT8 quantization** for deployment
4. **Enable custom CUDA kernels** for best performance
5. **Batch inputs** when possible for higher throughput

## 🐛 Issues?

If you encounter any problems:
1. Check [GitHub Issues](https://github.com/username/adaptive-hybrid-compression/issues)
2. Read [CONTRIBUTING.md](CONTRIBUTING.md)
3. Open a new issue with details

---

**Happy coding!** 🚀
