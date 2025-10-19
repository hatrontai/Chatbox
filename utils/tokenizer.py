import nltk

class tokenizer:
    def __init__(self, text: str) -> None:
        self.text = text
        
        tokens = self.split(self.text)
        self.vocab = list(sorted(list(set(tokens))))
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def split(self, text: str) -> list:
        lines = text.split('\n')
        tokens = []
        for line in lines:
            words = nltk.word_tokenize(line)
            tokens.extend(words)
            tokens.append('\n')
        tokens = tokens[:-1]  # Remove the last added newline
        return tokens
    
    def encode(self, text: str) -> list:
        tokens = self.split(text)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids
    
    def decode(self, token_ids: list) -> str:
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        text = ' '.join(tokens)
        return text

with open('./rawdata//data.txt', 'r') as file:
    data = file.read()

ENCODER = tokenizer(data, vocab=None)

if __name__ == "__main__":
    sample_text = "hi , how are you ?"
    print("size of vocab:", len(ENCODER.vocab))
    encoded = ENCODER.encode(sample_text)
    print("Encoded:", encoded)
    decoded = ENCODER.decode(encoded)
    print("Decoded:", decoded)
