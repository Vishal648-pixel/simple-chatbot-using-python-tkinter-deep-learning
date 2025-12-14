🤖 AI Chatbot (Python + Tkinter + Deep Learning)

This project is a Dark Mode AI Chatbot built using Python, NLTK, TensorFlow (Keras) and Tkinter GUI.
The chatbot is trained using an intents JSON file and can answer predefined questions.

📁 Project Structure
Chatbot-using-Python-master/
│
├── chatgui.py          # GUI application (Dark Mode)
├── train_chatbot.py    # Train the chatbot model
├── intents.json        # Training data (questions & responses)
├── model.h5            # Trained deep learning model
├── words.pkl           # Vocabulary
├── classes.pkl         # Intent classes
└── README.md

🧰 Requirements

Make sure you have:

Python 3.9 – 3.12

pip (comes with Python)

Internet connection (for first-time downloads)

🟢 STEP 1: Install Python

Download Python from:
👉 https://www.python.org/downloads/

While installing:
✅ Check “Add Python to PATH”
✅ Click Install Now

Verify installation:

python --version

🟢 STEP 2: Open Command Prompt (CMD)

Press Windows + R

Type cmd

Press Enter

🟢 STEP 3: Go to Project Folder
cd C:\Users\Vishal\Downloads\Chatbot-using-Python-master\Chatbot-using-Python-master


(Replace path if your folder is in a different location)

🟢 STEP 4: Install Required Python Packages

Run these commands one by one:

pip install numpy
pip install nltk
pip install tensorflow
pip install keras


⚠️ TensorFlow installation may take some time — please wait.

🟢 STEP 5: Download NLTK Data

Open Python shell:

python


Then run:

import nltk
nltk.download('punkt')
nltk.download('wordnet')
exit()

🟢 STEP 6: Train the Chatbot (Only Once)

This will create:

model.h5

words.pkl

classes.pkl

python train_chatbot.py


✅ After successful training, you will see accuracy & loss output.

🟢 STEP 7: Run the Chatbot GUI
python chatgui.py


💬 Sample Questions You Can Ask
Hi
Hello
What help you provide?
Find pharmacy
Open blood pressure module
Thanks
Bye

🛠 Common Issues & Fixes
❌ GUI takes time to open

✔ Model loads in background — please wait 2–5 seconds

❌ nltk resource not found
python -c "import nltk; nltk.download('punkt')"

❌ model.h5 not found



note-it wil take 1-2 minutes to load gui becuase it takes time to load data


<img width="650" height="812" alt="Screenshot 2025-12-14 202755" src="https://github.com/user-attachments/assets/48b004e6-28fc-4888-b988-b3936614cb42" />

