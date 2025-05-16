// Navigation handling
document.querySelectorAll('.nav-links li').forEach(link => {
    link.addEventListener('click', () => {
        // Update active state
        document.querySelectorAll('.nav-links li').forEach(l => l.classList.remove('active'));
        link.classList.add('active');

        // Show corresponding page
        const pageId = link.getAttribute('data-page') + '-page';
        document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
        document.getElementById(pageId).classList.add('active');
    });
});

// File upload handling
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const resultsContainer = document.getElementById('results-container');

// Drag and drop functionality
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#2196F3';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#ccc';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#ccc';
    
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        alert('Invalid file type. Please upload an image.');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.classList.remove('hidden');
        // Hide the upload box
        const uploadBox = document.getElementById('drop-zone');
        if (uploadBox) uploadBox.classList.add('hidden');
        // Show the analyze button
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) analyzeBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

// Optional: Add a function to reset/replace the image
function resetUpload() {
    previewContainer.classList.add('hidden');
    previewImage.src = '';
    const uploadBox = document.getElementById('drop-zone');
    if (uploadBox) uploadBox.classList.remove('hidden');
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) analyzeBtn.classList.add('hidden');
    fileInput.value = '';
    // Hide the Upload New Image button
    const reloadBtn = document.getElementById('reload-image-btn');
    if (reloadBtn) reloadBtn.style.display = 'none';
}

function analyzeImage() {
    const fileInput = document.getElementById('file-input');
    if (!fileInput.files[0]) {
        alert('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show loading state
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = '<div class="loading">Analyzing image...</div>';
    resultsContainer.classList.remove('hidden');

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        displayResults(data);
        // Send result to patient record
        if (data.results && data.results.length > 0) {
            let patientId = localStorage.getItem('currentPatientId');
            let patientName = localStorage.getItem('currentPatientName');
            let doctorName = localStorage.getItem('currentDoctorName');
            if (!patientId || !patientName || !doctorName) {
                patientId = prompt('Enter Patient ID to save this analysis to their record:');
                patientName = prompt('Enter Patient Name:');
                doctorName = prompt('Enter Doctor Name:');
                if (patientId) localStorage.setItem('currentPatientId', patientId);
                if (patientName) localStorage.setItem('currentPatientName', patientName);
                if (doctorName) localStorage.setItem('currentDoctorName', doctorName);
            }
            if (patientId && patientName && doctorName) {
                const top = data.results[0];
                fetch('http://127.0.0.1:5000/api/records', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        patient_id: patientId,
                        patient_name: patientName,
                        doctor_name: doctorName,
                        diagnosis: top.disease,
                        treatment: '',
                        confidence: top.confidence
                    })
                })
                .then(res => res.json())
                .then(recordData => {
                    console.log('Record saved:', recordData);
                    showPatientRecord(patientId, patientName, doctorName);
                    refreshAccessLogs();
                })
                .catch(err => {
                    console.error('Failed to save record:', err);
                });
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultsContainer.innerHTML = `<div class="error">Analysis failed: ${error.message}</div>`;
    });
}

function sendImageForAnalysis(imageData) {
    const formData = new FormData();
    formData.append('file', imageData);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error analyzing image: ' + error.message);
    });
}

// Updated displayResults function to match backend response
function displayResults(data) {
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.classList.remove('hidden');
    resultsContainer.innerHTML = '';

    if (data.results && data.results.length > 0) {
        // Show the top prediction in the main result card
        const top = data.results[0];
        const advice = generateMedicalAdvice(top.disease);
        const resultDiv = document.createElement('div');
        resultDiv.className = 'analysis-results';
        resultDiv.innerHTML = `
            <h3>Analysis Results</h3>
            <div class="result-item">
                <strong>Condition:</strong> ${top.disease}
            </div>
            <div class="result-item">
                <strong>Confidence:</strong> ${top.confidence.toFixed(2)}%
            </div>
            <div class="result-item diagnosis">
                <strong>Diagnosis:</strong> ${advice.description}
            </div>
        `;
        resultsContainer.appendChild(resultDiv);
        // Show the Upload New Image button
        const reloadBtn = document.getElementById('reload-image-btn');
        if (reloadBtn) reloadBtn.style.display = 'inline-block';

        // Show other predictions if they exist
        if (data.results.length > 1) {
            const othersDiv = document.createElement('div');
            othersDiv.className = 'other-results';
            othersDiv.innerHTML = '<h4>Also could be:</h4>';
            const ul = document.createElement('ul');
            data.results.slice(1).forEach(result => {
                const li = document.createElement('li');
                const advice = generateMedicalAdvice(result.disease);
                li.innerHTML = `<strong>${result.disease}:</strong> ${result.confidence.toFixed(2)}%<br><span class="diagnosis">${advice.description}</span>`;
                ul.appendChild(li);
            });
            othersDiv.appendChild(ul);
            resultsContainer.appendChild(othersDiv);
        }

        // Show medical advice for the top prediction
        displayMedicalAdvice(top.disease);
    } else {
        resultsContainer.innerHTML = '<p class="error">No prediction available</p>';
    }
    // Scroll to the results container so it's visible above the chatbox
    if (resultsContainer) {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
}

// Mock analysis function
function generateMedicalAdvice(condition) {
    const adviceMap = {
        'cavity': {
            description: 'A cavity is a small hole in the tooth caused by decay. It occurs when bacteria in your mouth produce acids that eat away at the tooth enamel.',
            advice: [
                'Visit a dentist for proper treatment',
                'Maintain good oral hygiene',
                'Use fluoride toothpaste',
                'Limit sugary foods and drinks'
            ]
        },
        'gingivitis': {
            description: 'Gingivitis is a common and mild form of gum disease that causes irritation, redness and swelling of your gingiva.',
            advice: [
                'Brush teeth twice daily',
                'Floss regularly',
                'Use antiseptic mouthwash',
                'Schedule professional cleaning'
            ]
        },
        // Add more conditions as needed
    };

    return adviceMap[condition.toLowerCase()] || {
        description: 'No specific information available for this condition.',
        advice: ['Please consult with a dental professional for proper diagnosis and treatment']
    };
}

function displayMedicalAdvice(condition) {
    const advice = generateMedicalAdvice(condition);
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return; // Prevent null reference error
    // Clear previous messages
    chatMessages.innerHTML = '';
    // Add description
    const descElement = document.createElement('div');
    descElement.className = 'chat-message bot';
    descElement.textContent = `Medical Description: ${advice.description}`;
    chatMessages.appendChild(descElement);
    // Add advice
    advice.advice.forEach(item => {
        const adviceElement = document.createElement('div');
        adviceElement.className = 'chat-message bot';
        adviceElement.textContent = `💡 ${item}`;
        chatMessages.appendChild(adviceElement);
    });
    // Show chatbot
    const chatbotContainer = document.getElementById('chatbot-container');
    if (chatbotContainer) {
        chatbotContainer.classList.remove('hidden');
    }
}

// Add event listener for file input change
document.getElementById('file-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview-image').src = e.target.result;
            document.getElementById('preview-container').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
});

// Mock patient search
function searchPatient() {
    const patientId = document.getElementById('patient-search').value;
    const patientInfo = document.getElementById('patient-info');
    
    if (!patientId) {
        alert('Please enter a patient ID');
        return;
    }

    // Make API call to backend
    fetch(`http://127.0.0.1:5000/patient/${patientId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Patient not found');
            }
            return response.json();
        })
        .then(data => {
            patientInfo.classList.remove('hidden');
            patientInfo.innerHTML = `
                <div class="patient-card">
                    <h3>Patient Information</h3>
                    <p><strong>Name:</strong> ${data.first_name} ${data.last_name}</p>
                    <p><strong>Date of Birth:</strong> ${data.date_of_birth}</p>
                    <p><strong>Gender:</strong> ${data.gender}</p>
                    <p><strong>Phone:</strong> ${data.phone_number}</p>
                    <p><strong>Email:</strong> ${data.email}</p>
                </div>
                <div class="medical-records">
                    <h3>Medical Records</h3>
                    <div class="records-list">
                        ${data.medical_records ? data.medical_records.map(record => `
                            <div class="record-item">
                                <p><strong>Visit Date:</strong> ${record.visit_date}</p>
                                <p><strong>Diagnosis:</strong> ${record.diagnosis}</p>
                                <p><strong>Treatment:</strong> ${record.treatment}</p>
                                ${record.xray_image_path ? `<img src="${record.xray_image_path}" alt="X-Ray Image">` : ''}
                            </div>
                        `).join('') : '<p>No medical records found</p>'}
                    </div>
                </div>
            `;
        })
        .catch(error => {
            patientInfo.innerHTML = `<p class="error">${error.message}</p>`;
            patientInfo.classList.remove('hidden');
        });
}

// On DOMContentLoaded, refresh logs from backend
document.addEventListener('DOMContentLoaded', function() {
    refreshAccessLogs();
});

function refreshAccessLogs() {
    fetch('http://127.0.0.1:5000/api/logs')
        .then(res => res.json())
        .then(logs => {
            const logsBody = document.getElementById('logs-body');
            logsBody.innerHTML = logs.map(log => `
                <tr>
                    <td>${log.timestamp}</td>
                    <td>${log.doctor_name}</td>
                    <td>${log.patient_name}</td>
                    <td>${log.patient_id}</td>
                    <td>${log.action}</td>
                    <td>${log.details}</td>
                </tr>
            `).join('');
        });
}

// Add these functions to your existing JavaScript

function toggleChat() {
    const chatMessages = document.querySelector('.chat-messages');
    const chatInput = document.querySelector('.chat-input');
    const minimizeBtn = document.getElementById('minimize-chat');
    
    if (chatMessages.style.display === 'none') {
        chatMessages.style.display = 'block';
        chatInput.style.display = 'flex';
        minimizeBtn.textContent = '-';
    } else {
        chatMessages.style.display = 'none';
        chatInput.style.display = 'none';
        minimizeBtn.textContent = '+';
    }
}

// Enhanced chatbot functionality
const chatHistory = [];

async function sendMessage() {
    const userInput = document.getElementById('user-message');
    const msg = userInput.value.trim();
    if (!msg) return;
    addChatMessage(msg, 'user');
    userInput.value = '';

    // Send to your Flask AI backend
    try {
        const response = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        const data = await response.json();
        addChatMessage(data.response, 'bot');
    } catch (error) {
        addChatMessage("Sorry, I couldn't connect to the AI server.", 'bot');
    }
}

function addChatMessage(text, sender) {
    const bar = document.getElementById('chat-messages-bar');
    const div = document.createElement('div');
    div.className = 'chat-message' + (sender === 'user' ? ' user' : '');
    div.textContent = text;
    bar.appendChild(div);
    bar.scrollTop = bar.scrollHeight;
}

// Add event listener for Enter key in chat input
document.getElementById('user-message').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});


function switchTab(tab) {
    document.querySelectorAll('.auth-tab').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.auth-form').forEach(form => form.classList.remove('active'));

    if (tab === 'login') {
      document.querySelector('.auth-tab:nth-child(1)').classList.add('active');
      document.getElementById('login-form').classList.add('active');
    } else {
      document.querySelector('.auth-tab:nth-child(2)').classList.add('active');
      document.getElementById('signup-form').classList.add('active');
    }
  }

  function login() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    if (!email || !password) {
        alert("Please enter both email and password.");
        return;
    }

    fetch('http://127.0.0.1:5000/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            email: email,
            password: password
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => { throw new Error(data.error || 'Login failed'); });
        }
        return response.json();
    })
    .then(data => {
        // Store user type in localStorage
        if (data.is_doctor) {
            localStorage.setItem('userType', 'doctor');
            window.location.href = 'doctor.html';
        } else {
            localStorage.setItem('userType', 'patient');
            window.location.href = 'index.html';
        }
    })
    .catch(error => {
        alert('Login failed: ' + error.message);
    });
}

function signup() {
    const name = document.getElementById('signup-name').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    const userType = document.querySelector('input[name="user-type"]:checked').value;

    fetch('http://127.0.0.1:5000/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: name,
            email: email,
            password: password,
            user_type: userType
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => { throw new Error(data.error || 'Signup failed'); });
        }
        return response.json();
    })
    .then(data => {
        localStorage.setItem('userType', userType);
        if (userType === 'doctor') {
            window.location.href = 'doctor.html';
        } else {
            window.location.href = 'index.html';
        }
    })
    .catch(error => {
        alert('Signup failed: ' + error.message);
    });
}

document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('file-input');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');
        const data = await response.json();
        // Handle response data
    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed. Please try again.');
    }
});


// Typing indicators
function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'chat-message bot typing';
    typingIndicator.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    chatMessages.appendChild(typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.querySelector('.typing');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Voice input/output
function startVoiceInput() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = document.getElementById('language-select').value;
    recognition.start();
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('user-message').value = transcript;
    };
}

function speakResponse(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = document.getElementById('language-select').value;
    window.speechSynthesis.speak(utterance);
}

// Medical knowledge base
const medicalKnowledgeBase = {
    "cavity": {
        "description": "Tooth decay caused by bacterial infection",
        "treatment": ["Fillings", "Crowns", "Root canals"],
        "prevention": ["Regular brushing", "Flossing", "Dental checkups"]
    },
    "gingivitis": {
        "description": "Inflammation of the gums",
        "treatment": ["Professional cleaning", "Antibacterial mouthwash"],
        "prevention": ["Good oral hygiene", "Regular dental visits"]
    }
    // Add more medical conditions
};

function getMedicalInfo(condition) {
    return medicalKnowledgeBase[condition.toLowerCase()] || {
        description: "Condition not found in knowledge base",
        treatment: ["Consult a medical professional"],
        prevention: []
    };
}

// Enhanced message handling
function handleUserMessage(message) {
    showTypingIndicator();
    
    // Check if message is a medical query
    const medicalInfo = getMedicalInfo(message);
    if (medicalInfo) {
        const response = `Medical Information:\nDescription: ${medicalInfo.description}\nTreatment: ${medicalInfo.treatment.join(', ')}\nPrevention: ${medicalInfo.prevention.join(', ')}`;
        addBotMessage(response);
        speakResponse(response);
    } else {
        // Handle regular conversation
        addBotMessage(message);
    }
    
    hideTypingIndicator();
}

// AI Assistant functionality
document.addEventListener('DOMContentLoaded', function() {
    const chatbotContainer = document.getElementById('chatbot-container');
    const openChatbotBtn = document.getElementById('open-chatbot');
    const closeChatbotBtn = document.getElementById('chat-close');
    const minimizeChatbotBtn = document.getElementById('chat-minimize');
    const maximizeChatbotBtn = document.getElementById('chat-maximize');

    // Open chatbot
    openChatbotBtn.addEventListener('click', function() {
        chatbotContainer.classList.remove('hidden');
    });

    // Close chatbot
    closeChatbotBtn.addEventListener('click', function() {
        chatbotContainer.classList.add('hidden');
    });

    // Minimize chatbot
    minimizeChatbotBtn.addEventListener('click', function() {
        chatbotContainer.classList.add('minimized');
    });

    // Maximize chatbot
    maximizeChatbotBtn.addEventListener('click', function() {
        chatbotContainer.classList.remove('minimized');
        chatbotContainer.classList.toggle('fullscreen');
    });

    // Send message function
    window.sendMessage = function() {
        const messageInput = document.getElementById('user-message');
        const messagesContainer = document.getElementById('chat-messages');
        const message = messageInput.value.trim();
        
        if (message) {
            // Add user message
            const userMessageDiv = document.createElement('div');
            userMessageDiv.className = 'chat-message user';
            userMessageDiv.textContent = message;
            messagesContainer.appendChild(userMessageDiv);
            
            // Clear input
            messageInput.value = '';
            
            // Simulate bot response (replace with actual AI response logic)
            setTimeout(() => {
                const botMessageDiv = document.createElement('div');
                botMessageDiv.className = 'chat-message bot';
                botMessageDiv.textContent = "I'm here to help with your dental queries!";
                messagesContainer.appendChild(botMessageDiv);
                
                // Scroll to bottom
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }, 1000);
        }
    };

    // Handle Enter key in message input
    document.getElementById('user-message').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    const reloadBtn = document.getElementById('reload-image-btn');
    if (reloadBtn) reloadBtn.onclick = resetUpload;
});

// Add this event listener
const analyzeBtn = document.querySelector('.analyze-btn');
if (analyzeBtn) {
    analyzeBtn.addEventListener('click', analyzeImage);
}

// Update send button event
const sendBtn = document.getElementById('send-btn');
if (sendBtn) sendBtn.onclick = sendMessage;

// Fullscreen toggle
const fullscreenBtn = document.getElementById('fullscreen-btn');
if (fullscreenBtn) {
    fullscreenBtn.onclick = function() {
        document.getElementById('chatbot-container').classList.toggle('fullscreen');
    };
}
const closeChatBtn = document.getElementById('close-chat');
if (closeChatBtn) {
    closeChatBtn.onclick = function() {
        document.getElementById('chatbot-container').classList.add('hidden');
    };
}

// Hide analyze button when clearing image (if you have a clear/reset function, add this logic there)

// Ensure AI Assistant button shows the chatbot
const openChatbotBtn = document.getElementById('open-chatbot');
const chatbotContainer = document.getElementById('chatbot-container');
if (openChatbotBtn && chatbotContainer) {
    openChatbotBtn.addEventListener('click', function() {
        chatbotContainer.classList.remove('hidden');
    });
}

async function askDentalAI(question) {
    const response = await fetch('http://127.0.0.1:5000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question })
    });
    if (!response.ok) throw new Error('Network response was not ok');
    const data = await response.json();
    return data.response;
}

// Update showPatientRecord to display all three fields
function showPatientRecord(patientId, patientName, doctorName) {
    const patientInfo = document.getElementById('patient-info');
    if (!patientInfo) return;
    patientInfo.classList.remove('hidden');
    patientInfo.innerHTML = `
        <div class="patient-card">
            <h3>Patient Information</h3>
            <p><strong>ID:</strong> ${patientId}</p>
            <p><strong>Name:</strong> ${patientName}</p>
            <p><strong>Doctor:</strong> ${doctorName}</p>
        </div>
    `;
}
