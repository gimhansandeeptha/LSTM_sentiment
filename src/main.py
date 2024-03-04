import torch
import torch.nn as nn
from embedding import GloveEmbedding
from model import LSTMClassifier

def create_embedding(weight_dict, trainable=False):
    vocab_size, embedding_dim = len(weight_dict), len(next(iter(weight_dict.values())))

    # Initialize an embedding layer with random weights
    emb_layer = nn.Embedding(vocab_size, embedding_dim)

    # Load the pre-trained weights into the embedding layer
    emb_layer.weight.data.copy_(torch.tensor(list(weight_dict.values())))

    # Set requires_grad based on the trainable flag
    emb_layer.weight.requires_grad = trainable

    return emb_layer


glove_embedding = GloveEmbedding()
vectors, words, word2idx, glove = glove_embedding.get_embedding()

embedding_layer = create_embedding(glove)
# weights = embedding_layer.weight.data

saved_model_path = "D:\Gimhan Sandeeptha\Gimhan\Sentiment-Email\LSTM_Production\Models\lstm_imdb_model.pth"
model = LSTMClassifier(embedding_layer,hidden_size=128)
model.load_state_dict(torch.load(saved_model_path))

model.eval()  # Set the model to evaluation mode

# Function for inferencing
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

# Example usage
text_to_predict = "This movie is great with lot of interesting scenes. really enjoyed the whole film. thank you"
predicted_label = predict_sentiment(text_to_predict)
print(f"predicted_label is: {predicted_label}")