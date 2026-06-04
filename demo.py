"""
Interactive demo for the from-scratch LLaMA 2 model.
Trains on a small corpus, then lets you type prompts and see continuations.
"""

from Math.Tensor import Tensor
from SequenceProcessing.Parameters.Llama2Parameter import Llama2Parameter
from SequenceProcessing.Classification.Llama2 import Llama2
from SequenceProcessing.Tokenizer.SimpleTokenizer import SimpleTokenizer


def main():
    # Bohemian rhapsody lyrics
    corpus: list[str] = """
Is this the real life?
Is this just fantasy?
Caught in a landslide
No escape from reality
Open your eyes
Look up to the skies and see
I'm just a poor boy
I need no sympathy
Because I'm easy come, easy go
Little high, little low
Any way the wind blows doesn't really matter to me, to me
Mama, just killed a man
Put a gun against his head
Pulled my trigger, now he's dead
Mama, life had just begun
But now I've gone and thrown it all away
Mama, ooh
Didn't mean to make you cry
If I'm not back again this time tomorrow
Carry on, carry on
As if nothing really matters
Too late, my time has come
Sends shivers down my spine
Body's aching all the time
Goodbye everybody, I've got to go
Gotta leave you all behind and face the truth
Mama, ooh (any way the wind blows)
I don't wanna die
I sometimes wish I'd never been born at all
I see a little silhouetto of a man
Scaramouche, Scaramouche, will you do the Fandango?
Thunderbolt and lightning very very frightening me
Gallileo, Gallileo
Gallileo, Gallileo
Gallileo Figaro, magnifico
I'm just a poor boy, nobody loves me
He's just a poor boy from a poor family
Spare him his life from this monstrosity
Easy come easy go, will you let me go?
Bismillah! No, we will not let you go (let him go)
Bismillah! We will not let you go (let him go)
Bismillah! We will not let you go (let me go)
Will not let you go (let me go)
Never, never, never, never let me go
No, no, no, no, no, no, no
Oh, mama mia, mama mia
Mama mia, let me go
Beelzebub has a devil put aside for me
For me
For me
So you think you can stone me and spit in my eye?
So you think you can love me and leave me to die?
Oh, baby
Can't do this to me, baby
Just gotta get out, just gotta get right outta here
Nothing really matters
Anyone can see
Nothing really matters
Nothing really matters to me
Any way the wind blows
        """.split("\n")

    print("[1/3] Building tokenizer...")
    tokenizer = SimpleTokenizer()
    for sentence in corpus:
        tokenizer.fit(sentence)
    vocab_size = tokenizer.getVocabularySize()
    print(f"       Vocabulary size: {vocab_size} tokens")
    print()

    print("[2/3] Preparing training data...")
    train_data = []
    for sentence in corpus:
        ids = tokenizer.encode(sentence)
        train_data.append(Tensor([float(t) for t in ids], (len(ids),)))
    print(f"       {len(train_data)} training sequences")
    print()

    print("[3/3] Training model (this may take a few minutes)...")
    param = Llama2Parameter.tinyLlama2(epoch=2, vocabulary_length=vocab_size)
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
