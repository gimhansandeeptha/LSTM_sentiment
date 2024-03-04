import bentoml
import torch
from embedding import GloveEmbedding


lstm_runner = bentoml.pytorch.get("lstm_model:latest").to_runner()
lstm_runner.init_local()


glove_embedding = GloveEmbedding()
vectors, words, word2idx, glove = glove_embedding.get_embedding()

# model.eval()  # Set the model to evaluation mode

# Function for inferencing
def preprocess(text):
    # Tokenize and convert text to indices
    tokens = text.split()
    indexed_tokens = [word2idx.get(word, 0) for word in tokens]

    # Convert indices to a PyTorch tensor
    input_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

    return input_tensor

# Example usage
text_to_predict = "This movie is great with lot of interesting scenes. really enjoyed the whole film. thank you"
preprocessed_text = preprocess(text_to_predict)
result  = lstm_runner.run(preprocessed_text)
print(result)