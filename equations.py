# 1. Define synthetic dataset with input values 'x' and unique 'delta' parameters
input_data = [
    {'id': 0, 'x': 10.0, 'delta': 0.5},
    {'id': 1, 'x': 20.0, 'delta': 1.2},
    {'id': 2, 'x': 30.0, 'delta': -0.8},
    {'id': 3, 'x': 40.0, 'delta': 2.5},
    {'id': 4, 'x': 50.0, 'delta': 0.1}
]

# 2. & 3. Calculate equations and store them in a structured trace
equation_trace = []
for entry in input_data:
    y_val = local_linear_equation(entry['x'], entry['delta'])
    trace_entry = {
        'entry_id': entry['id'],
        'x': entry['x'],
        'delta': entry['delta'],
        'equation_output': y_val,
        'description': f"y = {entry['x']} + {entry['delta']}"
    }
    equation_trace.append(trace_entry)

# 4. Print the trace to verify unique relationships
print('--- Equation Trace ---')
for item in equation_trace:
    print(f"ID {item['entry_id']}: {item['description']} => Result: {item['equation_output']}")
