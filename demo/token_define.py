# 1. Define a list of string tokens representing a sequence
tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

# 2. Create a vocabulary dictionary mapping each unique token to a unique integer ID
unique_tokens = sorted(list(set(tokens)))
vocab = {token: i for i, token in enumerate(unique_tokens)}

# 3. Convert the original sequence into a list of corresponding token IDs
token_ids = [vocab[token] for token in tokens]

# 4. Store and display the mapping and sequence for the pipeline
print(f'Vocabulary Mapping: {vocab}')
print(f'Original Tokens: {tokens}')
print(f'Token IDs: {token_ids}')
