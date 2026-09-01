# mini-llm-runtime

A minimal LLM inference runtime written from scratch in C++.

The goal of this project is to understand how Transformer inference works internally by implementing the main components manually instead of relying on PyTorch or an existing inference engine.

The current runtime can load and run **SmolLM2-135M** and generate text on CPU.

## Current pipeline

```text
text
  ↓
Byte-level BPE tokenizer
  ↓
token IDs
  ↓
embeddings
  ↓
30 Transformer blocks
  ↓
final RMSNorm
  ↓
logits
  ↓
greedy decoding
  ↓
generated text

Implemented
Safetensors model loading
BF16 → FP32 weight conversion
Token embeddings
RMSNorm
Grouped Query Attention (GQA)
9 query heads
3 key/value heads
Causal self-attention
RoPE positional embeddings
Attention output projection
SwiGLU MLP
Residual connections
30-layer Transformer forward pass
Tied embedding output projection
Byte-level BPE tokenizer
Greedy autoregressive generation
Model

Currently tested with:

HuggingFaceTB/SmolLM2-135M

Architecture:

hidden size:        576
layers:             30
attention heads:    9
KV heads:           3
head dimension:     64
MLP dimension:      1536
vocabulary:         49152
Validation

The C++ implementation was compared numerically against the Hugging Face reference implementation.

Intermediate Transformer block outputs and final logits match the reference implementation within normal FP32 numerical error.
ents:

C++20
CMake

Clone the repository and build:

cmake -S . -B build
cmake --build build

Run from the build directory:

cd build
./mini_llm

The model files are expected under:

models/SmolLM2-135M/
├── config.json
├── model.safetensors
├── vocab.json
└── merges.txt

Model weights are not included in the repository.

Project structure
include/
    Tensor.hpp
    attention.hpp
    loader.hpp
    mlp.hpp
    model.hpp
    ops.hpp
    tokenizer.hpp

src/
    attention.cpp
    loader.cpp
    main.cpp
    mlp.cpp
    model.cpp
    ops.cpp
    tokenizer.cpp
Next steps
KV cache
Proper Unicode tokenizer pre-tokenization
Temperature sampling
Top-k / top-p sampling
Memory profiling
Quantization
Faster matrix multiplication
SIMD optimizations
Continuous batching
CUDA backend
Why?

This project is primarily educational.
