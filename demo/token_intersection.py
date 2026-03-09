import torch

# 1. Initialize storage for intersections and the mapping trace
token_intersections = []
token_intersection_trace = []

# 2. Iterate through token_equation_trace to compare every unique pair (i, j)
n_tokens = len(token_equation_trace)
for i in range(n_tokens):
    for j in range(i + 1, n_tokens):
        entry_i = token_equation_trace[i]
        entry_j = token_equation_trace[j]

        # 3. Calculate the latent intersection value as the average of their outputs
        intersection_val = (entry_i['equation_output'] + entry_j['equation_output']) / 2.0

        # 4. Convert to a PyTorch tensor of shape (1,)
        intersection_tensor = torch.tensor([float(intersection_val)], dtype=torch.float32)
        token_intersections.append(intersection_tensor)

        # 5. Store metadata in the trace list
        token_intersection_trace.append({
            'pair': (i, j),
            'tokens': (entry_i['token'], entry_j['token']),
            'val_i': entry_i['equation_output'],
            'val_j': entry_j['equation_output'],
            'latent_intersection': intersection_val
        })

# 6. Print the total count and first few entries to verify
print(f'Total intersections generated: {len(token_intersections)}')
print('--- Token Intersection Trace (First 5) ---')
for item in token_intersection_trace[:5]:
    print(f"Pair {item['pair']} ({item['tokens'][0]}, {item['tokens'][1]}): Avg({item['val_i']}, {item['val_j']}) = {item['latent_intersection']}")
