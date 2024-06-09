# Build stage 
FROM python:3.9-slim-buster as build 

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY src/requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -i https://pypi.org/simple -r requirements.txt && \
    pip install --no-cache-dir -i https://pypi.org/simple --pre onnxruntime-genai && \
    pip install --no-cache-dir -i https://pypi.org/simple huggingface-hub

# Download the model using huggingface-cli
RUN huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir ./cpu_and_mobile

# Runtime stage
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the installed packages from the build stage
COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=build /app/cpu_and_mobile /app/cpu_and_mobile

# Copy the source code
COPY src/ .

# Set the command to run the main.py script
CMD ["python", "main.py"]