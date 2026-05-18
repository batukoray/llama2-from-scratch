"""
Interactive demo for the from-scratch LLaMA 2 model.
Trains on a small corpus, then lets you type prompts and see continuations.
"""

from Math.Tensor import Tensor
from SequenceProcessing.Parameters.Llama2Parameter import Llama2Parameter
from SequenceProcessing.Classification.Llama2 import Llama2
from SequenceProcessing.Tokenizer.SimpleTokenizer import SimpleTokenizer


def main():
    print("=" * 60)
    print("  LLaMA 2 From Scratch - Interactive Demo")
    print("  CS449 Introduction to Natural Language Processing")
    print("  Ozyegin University, Spring 2025-26")
    print("=" * 60)
    print()

    corpus = [
        "the cat sat on the mat the cat sat on the mat",
        "a dog ran in the park a dog ran in the park",
        "the bird flew over the tree the bird flew over the tree",
        "a fish swam in the lake a fish swam in the lake",
    ]

    print("[1/3] Building tokenizer...")
    tokenizer = SimpleTokenizer()
    for sentence in corpus:
        tokenizer.fit(sentence)
    print(f"       Vocabulary size: {tokenizer.getVocabularySize()} tokens")
    print()

    print("[2/3] Preparing training data...")
    train_data = []
    for sentence in corpus:
        ids = tokenizer.encode(sentence)
        train_data.append(Tensor([float(t) for t in ids], (len(ids),)))
    print(f"       {len(train_data)} training sequences")
    print()

    print("[3/3] Training model (this may take a few minutes)...")
    param = Llama2Parameter.tinyLlama2(epoch=3)
    model = Llama2(param)
    model.train(train_data)
    print("       Training complete!")
    print()

    print("-" * 60)
    print("Learned vocabulary:")
    for word in sorted(set(" ".join(corpus).split())):
        token_id = tokenizer.encode(word)[0]
        print(f"  {word:10s} -> {token_id}")
    print("-" * 60)
    print()

    print("Type a prompt to generate text. Type 'quit' to exit.")
    print("Try prompts like: 'the cat', 'a dog', 'the bird'")
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("Bye!")
            break

        prompt_ids = tokenizer.encode(prompt)

        unknown_tokens = [w for w in prompt.split() if w not in [tokenizer.decode([i]) for i in range(tokenizer.getVocabularySize())]]
        if unknown_tokens:
            print(f"  Warning: unknown words will be treated as <unk>: {unknown_tokens}")

        print(f"  Token IDs: {prompt_ids}")

        generated_ids = model.generateGreedy(prompt_ids, max_new_tokens=8)
        generated_text = tokenizer.decode(generated_ids)

        print(f"  Generated: {generated_text}")
        print()


if __name__ == "__main__":
    main()
