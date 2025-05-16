from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from flask_cors import CORS
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# 1. Load the GPT-2 tokenizer and model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add special tokens
tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    'bos_token': '[BOS]',
    'eos_token': '[EOS]'
})
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    # Add BOS and EOS tokens to each line
    texts = [f"[BOS]{text}[EOS]" for text in examples['text']]
    return tokenizer(texts, truncation=True, max_length=128, padding='max_length')

# 2. Load your dataset (each line is a Q&A pair)
dataset = load_dataset('text', data_files={'train': 'your_data.txt'})

# 3. Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# 4. Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-dental-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=100,
)

# 5. Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 6. Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
)

# 7. Train the model
print("Starting training...")
trainer.train()

# 8. Save the model and tokenizer
print("Saving model...")
trainer.save_model("./gpt2-dental-finetuned")
tokenizer.save_pretrained("./gpt2-dental-finetuned")

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.get_json()
    question = data.get('question') or data.get('message') or ''
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    prompt = f"Patient: {question}\n    AI:"
    response = generate_response(prompt)
    ai_response = response.split('AI:')[-1].strip()
    return jsonify({
        'question': question,
        'response': ai_response
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)

with app.app_context():
    db.create_all()

