# Priority 4C Classifier API - Apexkrieg Component

A machine learning API component of [Apexkrieg](https://apexkrieg.com) that classifies tasks and goals using the **Eisenhower Matrix** (also known as the **Urgent-Important Matrix**) methodology. The classifier uses a fine-tuned **DistilBERT** model optimized for performance with **PyTorch quantization**.

### 🚀 **Live Deployment**
The Priority 4C Classifier is deployed and accessible on **Hugging Face Spaces**:  
👉 **[Try it here](https://huggingface.co/spaces/Dev101LFG/AP4C)**

## 🏆 Priority Categories (Eisenhower Matrix)
The classifier categorizes tasks into the four quadrants of the Eisenhower Matrix:

| Quadrant | Description |
|----------|------------|
| **Q1** | Urgent and Important (**Do First**) |
| **Q2** | Not Urgent but Important (**Schedule**) |
| **Q3** | Urgent but Not Important (**Delegate**) |
| **Q4** | Not Urgent and Not Important (**Eliminate**) |

---
## ⚙️ Technical Stack
- **Framework**: Flask with ASGI support
- **ML Framework**: PyTorch
- **Model**: DistilBERT (Quantized)
- **Deployment**: Docker-ready
- **Runtime**: CUDA-compatible

## 🌐 API Endpoints

### 1️⃣ Health Check
**GET /**

📌 Returns a simple health check message.

### 2️⃣ Single Prediction
**POST /predict**

📌 Classifies a single task description.

#### **Request Example:**
```json
{
  "goal": "your task description here"
}
```

### 3️⃣ Batch Prediction
**POST /predict_batch**

📌 Classifies multiple task descriptions in a single request.

#### **Request Example:**
```json
[
  { "goal": "task 1" },
  { "goal": "task 2" },
  { "goal": "task 3" }
]
```

---
## 🔧 Environment Variables

- `PYTHONDONTWRITEBYTECODE`: Prevents Python from writing bytecode files.
- `PYTHONUNBUFFERED`: Ensures Python output is sent straight to the terminal.
- `HF_HOME`: HuggingFace models cache directory.
- `PORT`: Application port (default: **7860**).

---
## 🤖 Model Details
The system uses a fine-tuned **DistilBERT** model with the following specifications:

- **Vocabulary size**: 30,522 tokens
- **Maximum sequence length**: 512
- **Number of layers**: 6
- **Number of attention heads**: 12
- **Hidden dimension**: 768

---
## ⚡ Performance Optimizations
- **Dynamic quantization** using PyTorch.
- **Efficient batch processing** for lower latency.
- **Adaptive sequence length** based on input size.
- **Docker-optimized deployment**.
- **Non-root user** for enhanced security.

---
## 🛠 Development & Setup

### **Requirements**
- Python **3.10+**
- PyTorch **2.0+**
- HuggingFace Transformers
- Flask
- Docker
- CUDA-compatible GPU (for optimal performance)

### **Installation & Running Locally**
```bash
# Clone the repository
git clone https://github.com/DebBidhi/priority-4c-classifier.git
cd priority-4c-classifier

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

---
## 🚀 Deployment
The project is configured for deployment on **Hugging Face Spaces** using Docker. More details: [Hugging Face Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference)

---
## 🔒 Security Notes
- Runs as a **non-root user** in Docker for security.
- **Input validation and sanitization** to prevent attacks.
- **Memory-efficient model loading** to reduce resource usage.
- **Configurable resource limits** to optimize performance.

---
### 📜 License
This project is licensed under the **MIT License**.

---
### 🎯 **Apexkrieg - Empowering Intelligent Execution**
Priority 4C Classifier is a core component of **Apexkrieg**, helping users execute goals and tasks efficiently by leveraging **AI-driven tools**. 🚀🔥

