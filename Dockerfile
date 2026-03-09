FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the repository files into the container
COPY Manhattan-Network/ /app/

# Install dependencies
RUN pip install --no-cache-dir torch

# Run base: not demo
CMD python initialization.py && \
    python equations.py && \
    python intersections.py && \
    python neural.py && \
    python manhattan_algorithm.py && \

# Define the execution sequence for the NO-K pipeline
CMD python demo/token_define.py && \
    python demo/generate_equations.py && \
    python demo/token_intersection.py && \
    python demo/neural-manhattan.py && \
    python demo/predicted-token-output.py
