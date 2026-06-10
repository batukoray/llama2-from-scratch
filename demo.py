"""
Interactive demo for the from-scratch LLaMA 2 model.
Trains on a short literary corpus, then lets you type prompts and see continuations.
"""

from Math.Tensor import Tensor
from SequenceProcessing.Parameters.Llama2Parameter import Llama2Parameter
from SequenceProcessing.Classification.Llama2 import Llama2
from SequenceProcessing.Tokenizer.SimpleTokenizer import SimpleTokenizer


def main():
    # Short literary corpus for next-token generation.
    corpus: list[str] = """
It was the best of times and the worst of times and the model trained through the night until at last it could finish a sentence on its own.
The knight rode forth into the darkness armed with nothing but gradients determined to find the valley where all predictions would finally be correct.
    """.split("\n")
    corpus = [line.strip() for line in corpus if line.strip()]
    full_text = " ".join(corpus)

    print("[1/3] Building tokenizer")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(full_text)
    vocab_size = tokenizer.getVocabularySize()
    print(f"Vocabulary size: {vocab_size} tokens\n")

    print("[2/3] Preparing training data")
    param = Llama2Parameter.tinyLlama2(epoch=5, vocabulary_length=vocab_size)
    all_token_ids = tokenizer.encode(full_text)
    window_size = param.getContextLength() + 1
    stride = 4
    train_data = []
    for start in range(0, len(all_token_ids) - 1, stride):
        window_token_ids = all_token_ids[start:start + window_size]
        if len(window_token_ids) < 2:
            continue
        train_data.append(
            Tensor([float(token_id) for token_id in window_token_ids], (len(window_token_ids),))
        )
    print(f"       {len(train_data)} training sequences")
    print()

    print("[3/3] Training model (this may take a few minutes)...")
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
    print("Try prompts like: 'It was the best', 'The knight rode', 'armed with nothing'")
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

        generated_ids = model.generateSampled(prompt_ids, max_new_tokens=24, temperature=0.5)
        generated_text = tokenizer.decode(generated_ids)

        print(f"  Generated: {generated_text}")
        print()


if __name__ == "__main__":
    main()
