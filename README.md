# LLaMA 2 From Scratch

This repository is our implementation workspace for building a LLaMA 2-style decoder-only language model on top of the existing `SequenceProcessing` and `ComputationalGraph` primitives already used in this codebase.

The goal is not to reproduce Meta's full training pipeline or train a production 7B model on this machine. The goal is to implement the core LLaMA 2 architecture correctly, keep it aligned with the project's existing abstractions, and make it trainable on tiny toy datasets for validation.

## Project Goal

We will implement a LLaMA 2-style causal language model with:

- token embeddings
- decoder-only transformer blocks
- RMSNorm pre-normalization
- masked self-attention
- RoPE on query and key projections
- SwiGLU feed-forward layers
- residual connections
- final normalization and vocabulary projection
- autoregressive text generation

## Architecture Reference

Use this diagram as the architecture reference for the repository:

![LLaMA 2 architecture diagram](https://github.com/batukoray/assets_of_mine/blob/main/llama_diagram.png?raw=1)

## What Already Exists

This repository already gives us a useful starting point:

- [`SequenceProcessing/Classification/Transformer.py`](SequenceProcessing/Classification/Transformer.py) contains an existing computational-graph transformer implementation.
- [`SequenceProcessing/Parameters/TransformerParameter.py`](SequenceProcessing/Parameters/TransformerParameter.py) shows how model configuration is expressed in this project.
- [`SequenceProcessing/Functions`](SequenceProcessing/Functions) already includes helpers such as `Mean`, `Variance`, `SquareRoot`, `Inverse`, `Mask`, `Transpose`, and scalar transforms.
- [`test/TransformerTest.py`](test/TransformerTest.py) shows the current testing style.
- [`SequenceProcessing/README.md`](SequenceProcessing/README.md) documents the repository's coding conventions.

That means we should extend the current design instead of introducing a completely separate framework.

## Scope

### In scope

- implement a LLaMA 2-style model class
- add missing activation and positional functions
- support causal language modeling training
- support small-scale generation
- add tests for core math and graph construction
- document the architecture and usage

### Out of scope for the first milestone

- training a real 7B checkpoint
- distributed training
- full LLaMA tokenizer parity
- performance optimization for large-scale inference
- quantization, LoRA, or chat fine-tuning

## Implementation Plan

We will follow these phases in order.

### Phase 1: Lock the minimal architecture

Define the exact first-pass target so the implementation stays focused:

- decoder-only transformer
- causal mask only
- multi-head attention first
- grouped-query attention later if needed
- tiny configuration for tests
- architectural constants for LLaMA 2 7B kept as reference only

Deliverable:

- a stable design target for the first implementation pass

### Phase 2: Add a LLaMA 2 parameter class

Create a dedicated parameter object instead of overloading the current transformer parameter class too aggressively.

Planned file:

- `SequenceProcessing/Parameters/Llama2Parameter.py`

Responsibilities:

- vocabulary length
- embedding dimension
- decoder layer count
- attention head count
- optional key/value head count
- context length
- feed-forward dimension
- RMSNorm epsilon
- inherited optimizer, initialization, loss, epoch, and seed values

Helpful constructors:

- `tinyLlama2()` for unit tests
- `llama2_7B()` for architectural reference
- optional `llama2_13B()` for future extension

### Phase 3: Add missing primitives

We already have several useful tensor functions, but LLaMA 2 still needs a few missing pieces.

Planned files:

- `SequenceProcessing/Functions/SiLU.py`
- `SequenceProcessing/Functions/RotaryPositionEmbedding.py`

Possible helper:

- RMSNorm may be implemented as a graph-building helper instead of a standalone function if the current primitives are sufficient

Why this phase matters:

- SwiGLU requires SiLU
- LLaMA 2 attention requires RoPE on `Q` and `K`

### Phase 4: Implement the decoder-only model

Create a new model class instead of mutating the current encoder-decoder transformer into something it is not.

Planned file:

- `SequenceProcessing/Classification/Llama2.py`

Core responsibilities:

- input token handling
- token embedding lookup
- RMSNorm blocks
- masked self-attention
- RoPE application
- SwiGLU feed-forward block
- residual paths
- final normalization
- vocabulary projection
- training entrypoint
- generation entrypoint

High-level flow:

1. token ids
2. embeddings
3. repeated decoder layers
4. final RMSNorm
5. logits over vocabulary
6. softmax or sampling

### Phase 5: Add causal language modeling data preparation

The current transformer test setup is not yet shaped like autoregressive next-token prediction, so we need a clean LLaMA-style training path.

Planned behavior:

- input sequence: `[t0, t1, t2]`
- target sequence: `[t1, t2, t3]`
- left-to-right masking only
- sequence truncation to `context_length`

This phase should also define how token ids are converted into graph-compatible tensors.

### Phase 6: Add a minimal tokenizer

We do not need full tokenizer parity on day one, but we do need a dependable text-to-token path for tests and examples.

Planned file:

- `SequenceProcessing/Tokenizer/SimpleTokenizer.py`

First-pass goals:

- build a vocabulary from text
- encode text to ids
- decode ids back to text
- support `<unk>`, `<bos>`, and `<eos>`

Future upgrade path:

- replace or extend with a BPE-style tokenizer closer to the LLaMA 2 paper

### Phase 7: Testing and verification

We should prove correctness with a tiny model before adding more features.

Planned tests:

- `test/SiLUTest.py`
- `test/RotaryPositionEmbeddingTest.py`
- `test/Llama2Test.py`

Things to verify:

- SiLU matches expected values
- RoPE preserves shape and rotates values consistently
- causal masking blocks future positions
- tiny LLaMA 2 graph builds successfully
- forward pass returns logits with the expected shape
- probabilities are valid after softmax
- tiny model can overfit a very small sequence

Expected test command:

```bash
python3 -m unittest discover -s test -p "*Test.py"
```

### Phase 8: Documentation and packaging cleanup

Once the model works, we finish the repo integration work.

Updates likely needed:

- export the new modules in package `__init__` files
- update `setup.py` package list if new folders are added
- add a short usage example
- document practical limits of this implementation

## Proposed File Additions

The first implementation pass will likely add:

- `SequenceProcessing/Classification/Llama2.py`
- `SequenceProcessing/Parameters/Llama2Parameter.py`
- `SequenceProcessing/Functions/SiLU.py`
- `SequenceProcessing/Functions/RotaryPositionEmbedding.py`
- `SequenceProcessing/Tokenizer/SimpleTokenizer.py`
- `test/Llama2Test.py`
- `test/SiLUTest.py`
- `test/RotaryPositionEmbeddingTest.py`

## Definition of Done for Milestone 1

We will consider the first milestone complete when:

- a tiny LLaMA 2 model can be instantiated from a dedicated parameter class
- decoder-only masked self-attention works end to end
- RoPE and SwiGLU are implemented and tested
- the model can train on a toy next-token dataset
- the model can generate a short continuation from a prompt
- all new unit tests pass

## Notes and Constraints

- This codebase is centered around a custom computational graph, so correctness and clean graph construction matter more than raw speed.
- We should preserve the repository's existing style rules from `SequenceProcessing/README.md`.
- The practical implementation target is a small, testable model, even if we also expose LLaMA 2 7B dimensions as configuration presets.
- It is safer to add `Llama2.py` as a new model class than to heavily repurpose the current `Transformer.py`.

## Immediate Next Step

The next concrete step is to implement `Llama2Parameter.py`, `SiLU.py`, and `RotaryPositionEmbedding.py`, because those pieces define the contract the main `Llama2.py` model will depend on.
