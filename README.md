<img src="https://github.com/batukoray/assets_of_mine/blob/main/llama_banner.png?raw=1" alt="LLaMA 2 from Scratch" width="700">

This repository is our implementation workspace for building a LLaMA 2-style decoder-only language model on top of the existing `SequenceProcessing` and `ComputationalGraph` primitives already used in this codebase.

The goal is not to reproduce Meta's full training pipeline or train a production 7B model on this machine. The goal is to implement the core LLaMA 2 architecture correctly, keep it aligned with the project's existing abstractions, and make it trainable on tiny toy datasets for validation.

## Project Context

This repository is being developed as a course project for `CS449 Introduction to Natural Language Processing` at `Özyeğin University`.

## Project Team

- `Batu Koray Masak` - Sophomore, double major in AI and Data Engineering and Computer Science, Özyeğin University
- `Olcay Aras` - Sophomore, Computer Science, Özyeğin University

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

We are using this diagram as the architecture reference for the repository:

![LLaMA 2 architecture diagram](https://github.com/batukoray/assets_of_mine/blob/main/llama_diagram_black_white_rounded.png?raw=1)
