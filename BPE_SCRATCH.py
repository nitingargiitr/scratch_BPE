class BPE:
    def __init__(self):
        # mapping token id to token string 
        self.vocab = {}
        # mapping token string to token id
        self.inverse_vocab = {}
        # stores the merges which are made during training
        self.bpe_merges = {}
    
    
    def preprocess_text(self, text):
        # Remove unnecessary whitespaces
        text = text.strip()

        processed = []
        for char in text:
            if char == " ":
                processed.append("_")
            else:
                processed.append(char)
        return processed


    # initializing the vocab
    def initializing_vocab(self, processed_text):
        unique_chars = []
        
        # basic vocab using the ascii characters 
        unique_chars = [chr(i) for i in range(256)]

        # Add any additional characters from the processed text
        for char in processed_text:
            if char not in unique_chars:
                unique_chars.append(char)

        # Ensure the whitespace token is present
        if "_" not in unique_chars:
            unique_chars.append("_")

        # Create mappings
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}
        
    
    def find_freq_pair(self, token_ids):
        # count all adjacent pairs 
        pair_counts = {}
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            if pair in pair_counts:
                pair_counts[pair] = pair_counts[pair] + 1
            else:
                pair_counts[pair] = 1

        # finding the pair with the highest count
        max_pair = None
        max_count = 0
        for pair in pair_counts:
            if pair_counts[pair] > max_count:
                max_count = pair_counts[pair]
                max_pair = pair

        return max_pair  


    def replace_pair(self, token_ids, pair_to_merge, new_id):
        result = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair_to_merge:
                result.append(new_id)  # merge the pair into one new token
                i = i + 2  # skip the next token because it is part of the merged pair
            else:
                result.append(token_ids[i])  
                i = i + 1
        return result


    def train(self,text,vocab_size):
        processed_text = self.preprocess_text(text)
        self.initializing_vocab(processed_text)

        # convert processed text to token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        new_id = len(self.vocab)
        while new_id < vocab_size:
            pair = self.find_freq_pair(token_ids)
            if pair is None:
                break
            token_ids = self.replace_pair(token_ids, pair, new_id)
            self.bpe_merges[pair] = new_id
            new_id = new_id + 1

        # adding the new merged tokens to the vocab
        for (id1, id2), merge_id in self.bpe_merges.items():
            merged_token = self.vocab[id1] + self.vocab[id2]
            self.vocab[merge_id] = merged_token
            self.inverse_vocab[merged_token] = merge_id

    # printing the updated vocab
    def print_vocab(self):
        print("Vocabulary:")
        for token_id in sorted(self.vocab):
            print(f"{token_id}: {self.vocab[token_id]}")

    # to check what merges have been done
    def print_merges(self):
        print("BPE merges:")
        for pair, new_id in self.bpe_merges.items():
            print(f"{pair}" + " - " + f"{new_id}: {self.vocab[new_id]}")
    
    
    def tokenize(self, text):
        processed = self.preprocess_text(text)
        token_ids = [self.inverse_vocab[char] for char in processed]

        merge_steps = sorted(self.bpe_merges.items(), key=lambda x: x[1])
        for pair, new_id in merge_steps:
            token_ids = self.replace_pair(token_ids, pair, new_id)

        return [(token_id, self.vocab[token_id]) for token_id in token_ids]

    def encode(self, text):
        return [token_id for token_id, _ in self.tokenize(text)]

    def decode(self, token_ids):
        tokens = [self.vocab[token_id] for token_id in token_ids]
        return "".join(tokens).replace("_", " ")


bpe = BPE()
long_text = """
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. 
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed 
in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters.

The Time Traveller (for so it will be convenient to speak of him) was expounding a recondite matter to us. His grey eyes shone 
and twinkled, and his usually pale face was flushed and animated. The fire burned brightly, and the soft radiance of the incandescent 
lights in the lilies of silver caught the bubbles that flashed and passed with a flicker of green and golden fire.

Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular 
to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving 
off the spleen and regulating the circulation.

All human things are subject to decay, and when fate summons, monarchs must obey. Whether we fall by ambition, blood, or lust, 
like diamonds we are cut with our own dust.

Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the Galaxy lies a small unregarded 
yellow sun. Orbiting this at a distance of roughly ninety-two million miles is an utterly insignificant little blue green planet 
whose ape-descended life forms are so amazingly primitive that they still think digital watches are a pretty neat idea.

It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch 
of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, 
it was the winter of despair.
"""

bpe.train(long_text, vocab_size=600)
# bpe.print_vocab()
# bpe.print_merges()
text = "hello my name is nitin "
tokens = bpe.tokenize(text)
print("Tokens:", tokens)
print("IDs:", bpe.encode(text))
print("Decoded:", bpe.decode(bpe.encode(text)))




    