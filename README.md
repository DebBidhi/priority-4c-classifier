# Priority 4C Classifier API - Apexkrieg Component

A machine learning API component of [Apexkrieg](https://apexkrieg.com) that classifies tasks and goals using the Eisenhower Matrix (also known as the Urgent-Important Matrix) methodology. The classifier uses a fine-tuned DistilBERT model optimized for performance using PyTorch quantization.

## Priority Categories (Eisenhower Matrix)

The classifier categorizes tasks into the four quadrants of the Eisenhower Matrix:
- Q1: Urgent and Important (Do First)
- Q2: Not Urgent but Important (Schedule)
- Q3: Urgent but Not Important (Delegate)
- Q4: Not Urgent and Not Important (Eliminate)

## Technical Stack

- **Framework**: Flask with ASGI support
- **ML Framework**: PyTorch
- **Model**: DistilBERT (Quantized)
- **Deployment**: Docker-ready
- **Runtime**: CUDA-compatible

## API Endpoints

### 1. Health Check�
GET /
Returns a simple health check message.

### 2. Single Prediction
POST /predict
Content-Type: application/json
{
"goal": "your task description here"
}

### 3. Batch Prediction
POST /predict_batch
Content-Type: application/json
[
{ "goal": "task 1" },
{ "goal": "task 2" },
{ "goal": "task 3" }
]


## Environment Variables

- `PYTHONDONTWRITEBYTECODE`: Prevents Python from writing bytecode files
- `PYTHONUNBUFFERED`: Ensures Python output is sent straight to terminal
- `HF_HOME`: HuggingFace models cache directory
- `PORT`: Application port (default: 7860)

## Model Details

The system uses a fine-tuned DistilBERT model with the following specifications:
- Vocabulary size: 30,522 tokens
- Maximum sequence length: 512
- Number of layers: 6
- Number of attention heads: 12
- Hidden dimension: 768

## Performance Optimizations

- Dynamic quantization using PyTorch
- Efficient batch processing
- Adaptive sequence length based on input
- Docker-optimized deployment
- Non-root user for security

## Development

Requirements:
- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers
- Flask
- Docker
- CUDA-compatible GPU (for optimal performance)

## Deployment

The project is configured for deployment on HuggingFace Spaces with Docker support. See the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Security Notes

- Runs as non-root user in Docker
- Input validation and sanitization
- Memory-efficient model loading
- Configurable resource limits