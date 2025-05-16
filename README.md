🦷 AI-Powered Dental Diagnosis System

This project is a smart, web-based healthcare assistant that combines Computer Vision and Natural Language Processing (NLP) to deliver intelligent dental support. It allows users to:

- 🖼️ Upload dental images for automated disease detection using deep learning.
- 💬 Ask dental health-related questions via an AI chatbot powered by a fine-tuned GPT-2 model.
- 🖥️ Interact through a Flask-based web API, enabling modular, scalable, and educational use.

> 🔗 [Watch on YouTube](https://youtu.be/UuuptL5nYww)


📌 Key Features

- Image Classification: Detects conditions like cavities, gingivitis, ulcers, etc., from intraoral images using a CNN model (ResNet50V2).
- AI Chatbot: Uses GPT-2, fine-tuned on dental FAQs, to respond intelligently to dental health queries.
- RESTful API: Built using Flask with endpoints like /predict, /ask, and /api/logs.
- CORS-Enabled Communication: Allows smooth interaction between frontend (port 5500) and backend (port 5000).
- Logging System: Captures and serves logs for monitoring and debugging.
- Secure & Ethical: Trained only on anonymized or synthetic data; no real patient info is used.

🧰 Tech Stack

- Backend: Python, Flask, Flask-CORS, Flask-SQLAlchemy
- ML & NLP: TensorFlow, Keras, Hugging Face Transformers, GPT-2, PyTorch
- Utilities: NumPy, Pillow, Matplotlib, Seaborn, scikit-learn
- Model Training: Jupyter Notebooks for reproducibility

