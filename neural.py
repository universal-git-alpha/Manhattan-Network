torch.manual_seed(42)

# 1. Instantiate the NOKNeuralLayer with input_size=1 and alpha=0.1
nok_layer = NOKNeuralLayer(input_size=1, alpha=0.1)

# 2. Containers for corrected tensors and trace
corrected_intersections = []
neural_impact_trace = []

# 3. & 4. Iterate and perform forward pass
for idx, intersection_tensor in enumerate(latent_intersections):
    # Forward pass
    with torch.no_grad():
        corrected_tensor = nok_layer(intersection_tensor)
    
    # Calculate impact (output - input)
    input_val = intersection_tensor.item()
    output_val = corrected_tensor.item()
    impact = output_val - input_val
    
    # 5. Store results and update trace
    corrected_intersections.append(corrected_tensor)
    neural_impact_trace.append({
        'intersection_id': idx,
        'original_val': input_val,
        'corrected_val': output_val,
        'impact': impact
    })

# 6. Print the first few entries to verify scaling and corrections
print('--- Neural Impact Trace (First 5) ---')
for entry in neural_impact_trace[:5]:
    print(f"ID {entry['intersection_id']}: {entry['original_val']:.4f} -> {entry['corrected_val']:.4f} (Impact: {entry['impact']:.6f})")

print(f'\nProcessed {len(corrected_intersections)} intersections through the neural layer.')
