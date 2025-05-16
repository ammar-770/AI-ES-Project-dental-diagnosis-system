from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def generate_response(model, tokenizer, prompt, max_length=100):
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the fine-tuned model and tokenizer
model_path = "./gpt2-dental-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Sample questions to test
test_questions = [
    "What is a dental abscess?",
    "How do I prevent cavities?",
    "What are the symptoms of gingivitis?",
    "How often should I visit the dentist?",
    "What causes tooth sensitivity?"
]

print("Testing the fine-tuned model...\n")

for question in test_questions:
    prompt = f"Patient: {question}\n    AI:"
    response = generate_response(model, tokenizer, prompt)
    print(f"Question: {question}")
    print(f"Response: {response}\n") 