import torch
import torch.nn as nn
from embedding import GloveEmbedding
from model import LSTMClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)

def create_embedding(weight_dict, trainable=False):
    vocab_size, embedding_dim = len(weight_dict), len(next(iter(weight_dict.values())))

    # Initialize an embedding layer with random weights
    emb_layer = nn.Embedding(vocab_size, embedding_dim)

    # Load the pre-trained weights into the embedding layer
    emb_layer.weight.data.copy_(torch.tensor(list(weight_dict.values())))

    # Set requires_grad based on the trainable flag
    emb_layer.weight.requires_grad = trainable

    return emb_layer

def predict_sentiment(text):
    # Tokenize and convert text to indices
    tokens = text.split()
    indexed_tokens = [word2idx.get(word, 0) for word in tokens]

    # Convert indices to a PyTorch tensor
    input_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted sentiment
    predicted_sentiment = torch.argmax(output).item()

    return predicted_sentiment


# Load the trained PyTorch model
glove_embedding = GloveEmbedding()
vectors, words, word2idx, glove = glove_embedding.get_embedding()

embedding_layer = create_embedding(glove)
# weights = embedding_layer.weight.data

saved_model_path = "D:\Gimhan Sandeeptha\Gimhan\Sentiment-Email\LSTM_Production\Models\lstm_imdb_model.pth"
model = LSTMClassifier(embedding_layer,hidden_size=128)
model.load_state_dict(torch.load(saved_model_path))

model.eval()  # Set the model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        text = data['text']

        # Perform inference
        output = model_inference(text)

        # Return the result
        return jsonify({'sentiment': output})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def model_inference(text):
    # Implement your inference logic here
    # This could involve tokenization, encoding, and feeding data to the model
    # Example: output = model(text)
    predicted_label = predict_sentiment(text)
    return predicted_label

if __name__ == '__main__':
    app.run(debug=True)
