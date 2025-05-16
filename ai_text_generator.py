from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import time

app = Flask(__name__)
CORS(app)

# Load local model once
model_path = "./gpt2-dental-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
local_model = GPT2LMHeadModel.from_pretrained(model_path)

# OpenAI API key (use env variable for security)
openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-...")  # Replace with your key or set env var

def generate_local_response(prompt, max_length=40):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = local_model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=False,  # Deterministic and faster
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_openai_response(prompt, max_tokens=100):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a dental AI assistant. Answer patient questions and give advice about dental diseases."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()
    data = request.get_json()
    question = data.get('question') or data.get('message') or ''
    model_choice = data.get('model', 'local')  # 'local' or 'openai'
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    inference_start = time.time()
    if model_choice == 'openai':
        response = generate_openai_response(question)
    else:
        prompt = f"Patient: {question}\n    AI:"
        response = generate_local_response(prompt)
        response = response.split('AI:')[-1].strip()
    inference_end = time.time()

    print(f"[TIMING] Inference time: {inference_end - inference_start:.3f} seconds")
    print(f"[TIMING] Total /ask time: {inference_end - start_time:.3f} seconds")

    # Example: after model prediction
    results = [
        {"disease": "ulcer", "confidence": 99.86},
        {"disease": "cavity", "confidence": 0.10},
        {"disease": "healthy", "confidence": 0.04}
    ]
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5001)
