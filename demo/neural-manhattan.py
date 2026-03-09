torch.manual_seed(42)

# 1. Initialize the NOKNeuralLayer with specified parameters
token_nok_layer = NOKNeuralLayer(input_size=1, alpha=0.1)

# 2. & 3. Iterate through token_intersections and track corrections
token_corrected_intersections = []
token_neural_impact_trace = []

for idx, intersection_tensor in enumerate(token_intersections):
    with torch.no_grad():
        corrected_tensor = token_nok_layer(intersection_tensor)
    
    input_val = intersection_tensor.item()
    output_val = corrected_tensor.item()
    impact = output_val - input_val
    
    token_corrected_intersections.append(corrected_tensor)
    token_neural_impact_trace.append({
        'intersection_id': idx,
        'original_val': input_val,
        'corrected_val': output_val,
        'impact': impact
    })

# 4. Convert the list of corrected tensors into a single 1D PyTorch tensor
token_corrected_values = torch.cat(token_corrected_intersections).view(-1)

# 5. Apply the Manhattan aggregation formula: sum(|x_i - x_{i+1}|)
token_manhattan_components = torch.abs(token_corrected_values[:-1] - token_corrected_values[1:])
aggregated_manhattan_value = torch.sum(token_manhattan_components).item()

# 6. Print the result to verify
print(f'--- Token Neural Refinement & Aggregation ---')
print(f'Processed {len(token_corrected_intersections)} intersections.')
print(f'Final Aggregated Manhattan Value: {aggregated_manhattan_value:.6f}')

# Display a snippet of the impact trace
print('\nSample Impact (First 3):')
for entry in token_neural_impact_trace[:3]:
    print(f"ID {entry['intersection_id']}: {entry['original_val']:.2f} -> {entry['corrected_val']:.4f} (Impact: {entry['impact']:.6f})")
