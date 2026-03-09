token_equation_trace = []

for index, tid in enumerate(token_ids):
    # Using position (index) as x and token ID (tid) as delta
    y_val = local_linear_equation(index, tid)
    
    # Store the dictionary in the trace
    token_equation_trace.append({
        'token': tokens[index],
        'token_id': tid,
        'position': index,
        'equation_output': y_val,
        'description': f'y = {index} (pos) + {tid} (id)'
    })

print('--- Token Equation Trace (First 5) ---')
for item in token_equation_trace[:5]:
    print(f"Token: '{item['token']}' | Pos: {item['position']} | ID: {item['token_id']} | Result: {item['equation_output']}")
