# IMDb Sentiment Analysis with RNN  
This project is a simple yet effective **Sentiment Analysis app** built using a **Recurrent Neural Network (RNN)** to classify IMDb movie reviews as **Positive 🙂** or **Negative 😠**.

## Model Architecture  
- Embedding Dimension: 64  
- Hidden Dimension: 128  
- Output: Binary classification (Positive/Negative)  
- Framework: PyTorch  
- Dataset: IMDb Reviews  
- Training: Trained using cross-entropy loss and Adam optimizer

## Technologies Used  
- Python  
- PyTorch  
- TorchText  
- Streamlit  

## Project Structure  
```
.
├── app.py              # Streamlit app for inference  
├── model.pth           # Trained PyTorch RNN model  
├── vocab.pth           # Vocabulary object for tokenization  
├── requirements.txt    # Python dependencies  
└── README.md           # Project documentation  
```

## Setup Instructions  

### 1. Clone the Repository  
```bash  
git clone https://github.com/abinayasangeetha/sentiment-rnn-app.git  
cd sentiment-rnn-app  
```

### 2. Create Virtual Environment (Optional but Recommended)  
```bash  
python -m venv venv  
source venv/bin/activate       # On Windows: venv\Scripts\activate  
```

### 3. Install Dependencies  
```bash  
pip install -r requirements.txt  
```

### 4. Run the App  
```bash  
streamlit run app.py  
```

## How to Use  
1. Enter an IMDb-style movie review in the input box.  
2. Click the "Analyze Sentiment" button.  
3. The app will display whether the review is **Positive 🙂** or **Negative 😠**.

##  Example Inputs  
- **"This movie was fantastic! The acting was incredible."** → Positive 🙂  
- **"Terrible plot and poor performance. Not recommended."** → Negative 😠
## Output:
<img width="1885" height="974" alt="Screenshot 2025-07-31 115302" src="https://github.com/user-attachments/assets/a5bd383c-45b9-4d61-9c32-395b73f069a5" />

<img width="1876" height="899" alt="Screenshot 2025-07-31 115350" src="https://github.com/user-attachments/assets/2623ae64-5386-4b2f-a9b0-308c5e79c5bd" />


## 📄 License  
This project is open-source and available under the [MIT License](LICENSE).


