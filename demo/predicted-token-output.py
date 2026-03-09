import math

# 1. Calculate the 'predicted_token_id' by rounding the aggregated_manhattan_value
# We use the variable aggregated_manhattan_value from the previous step
raw_id = round(aggregated_manhattan_value)

# 2. Apply modulo operation using the length of the vocab to ensure it's in range
vocab_size = len(vocab)
predicted_token_id = raw_id % vocab_size

# 3. Create a reverse vocabulary mapping (ID to token string)
reverse_vocab = {v: k for k, v in vocab.items()}

# 4. Retrieve the 'predicted_token_string'
predicted_token_string = reverse_vocab[predicted_token_id]

# 5. Print the final results
print(f'--- Final Token Prediction Result ---')
print(f'Aggregated Manhattan Value: {aggregated_manhattan_value:.6f}')
print(f'Rounded Raw ID: {raw_id}')
print(f'Final Predicted Token ID (Modulo {vocab_size}): {predicted_token_id}')
print(f'Predicted Next Token: "{predicted_token_string}"')
