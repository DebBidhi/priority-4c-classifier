import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertConfig
from flask import Flask, request, jsonify
from asgiref.wsgi import WsgiToAsgi
import torch.quantization

app = Flask(__name__)

# Global variables
device = torch.device('cpu')
config = DistilBertConfig.from_json_file('config.json')
config.num_labels = 4
model_save_path = './P4C.pt'
tokenizer = DistilBertTokenizerFast(
    vocab_file='vocab.txt',
    special_tokens_map_file='special_tokens_map.json',
    tokenizer_config_file='tokenizer_config.json'
)

# Load and quantize model at startup
Finetuned_model = DistilBertForSequenceClassification(config)
Finetuned_model.load_state_dict(torch.load(model_save_path, map_location=device))
quantized_model = torch.quantization.quantize_dynamic(
    Finetuned_model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_model.to(device)
quantized_model.eval()  # Set the model to evaluation mode

categories = ['Urgent and Important', 'Not Urgent but Important', 'Urgent but Not Important', 'Not Urgent and Not Important']

def cached_predict_category(text, max_length):
    # Tokenize the input text
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # Get model prediction
    with torch.no_grad():
        outputs = quantized_model(**encoding)
        predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    
    return categories[predicted_class_idx]

def predict_categories(texts, max_length):
    encodings = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    with torch.no_grad():
        outputs = quantized_model(**encodings)
        predicted_class_idx = torch.argmax(outputs.logits, dim=1).tolist()
    
    return [categories[idx] for idx in predicted_class_idx]

@app.route('/')
def home():
    return "Hello, this is the P4 Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'goal' not in data:
        return jsonify({'error': 'Missing goal field'}), 400
    goal = data['goal']
    max_length = min(max(3 * len(goal.split()), 128), 256)  # Cap at 256
    predicted_category = cached_predict_category(goal, max_length)
    return jsonify({'predicted_category': predicted_category})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.json
    if 'goals' not in data:
        return jsonify({'error': 'Missing goals field'}), 400
    goals = data['goals']
    max_length = min(max(max(3 * len(goal.split()) for goal in goals), 128), 256)  # Cap at 256
    predicted_categories = predict_categories(goals, max_length)
    return jsonify({'predicted_categories': predicted_categories})

# Wrap the Flask app with the ASGI adapter
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # For development only. Use a production WSGI server in production!
    app.run(host='0.0.0.0', port=7860, debug=False)