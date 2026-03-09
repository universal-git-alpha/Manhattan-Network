import torch

# 1. & 4. Initialize storage for intersections and the mapping trace
latent_intersections = []
intersection_trace = []

# 2. Iterate through equation_trace to compare every unique pair (i, j)
n = len(equation_trace)
for i in range(n):
    for j in range(i + 1, n):
        entry_i = equation_trace[i]
        entry_j = equation_trace[j]
        
        # Since these are local linear representations y = x + delta,
        # we compute the latent consensus as the midpoint/average of their outputs
        # to find where these local relationships converge in the latent space.
        intersection_val = (entry_i['equation_output'] + entry_j['equation_output']) / 2.0
        
        # 3. Represent intersection as a PyTorch tensor for the neural layer
        intersection_tensor = torch.tensor([intersection_val], dtype=torch.float32)
        
        latent_intersections.append(intersection_tensor)
        
        # Store mapping trace
        intersection_trace.append({
            'pair': (entry_i['entry_id'], entry_j['entry_id']),
            'val_i': entry_i['equation_output'],
            'val_j': entry_j['equation_output'],
            'latent_intersection': intersection_val
        })

# 5. Print the calculated intersection points
print(f'--- Latent Intersection Trace (Total: {len(intersection_trace)}) ---')
for item in intersection_trace:
    print(f"Pair {item['pair']}: Avg({item['val_i']}, {item['val_j']}) = {item['latent_intersection']}")

print('\nLatent intersections prepared as tensors.')
