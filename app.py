import torch
import torch.nn as nn
import streamlit as st

# ----- Model Definition (same as Colab) -----
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# ----- Load Vocab and Model -----
vocab = torch.load("vocab.pth")  # Make sure it's the same vocab saved in Colab

vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
output_dim = 2

model = RNN(vocab_size, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("sentiment_model.pth", map_location=torch.device('cpu')))
model.eval()

# ----- Tokenizer -----
def tokenize(text):
    return [vocab[word] if word in vocab else vocab["<UNK>"] for word in text.lower().split()]

# ----- Streamlit UI -----
st.title("🎬 IMDB Sentiment Analysis with RNN")

input_text = st.text_input("Enter a movie review:")

if input_text:
    tokens = tokenize(input_text)
    input_tensor = torch.tensor(tokens).unsqueeze(0)  # shape: (1, seq_len)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # ✅ Use same label mapping as in Colab: 1 = Positive, 0 = Negative
    sentiment = "Positive 😀" if prediction == 1 else "Negative 😠"
    st.write(f"**Sentiment:** {sentiment}")
