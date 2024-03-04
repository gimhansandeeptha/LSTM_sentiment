import os
import zipfile
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer

from sklearn.model_selection import train_test_split

from collections import Counter
import re
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

folder_path = "D:/Gimhan Sandeeptha/Gimhan/Sentiment-Email/LSTM_Production/Datasets/"
extracted_folder = "glove"
file_name = "glove.6B.zip"
zip_file_path = os.path.join(folder_path, file_name)

if not os.path.exists(zip_file_path):
    print("Downloading files...")
    print('''Please fist download the vocab by following power shell command: \n 
          Invoke-WebRequest -Uri "http://nlp.stanford.edu/data/glove.6B.zip" -OutFile {zip_file_path}"''')
    sys.exit()

if not os.path.exists(os.path.join(folder_path, extracted_folder)):
    print("Extracting files...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(folder_path, extracted_folder))

print("Files are downloaded and extracted successfully")

glove_path = os.path.join(folder_path, extracted_folder)
glove_path


words = []
idx = 2
word2idx = {}
embedding_dim =100
vectors = []
if not os.path.exists(f'{glove_path}/6B.100_words.pkl'):
  with open(f'{glove_path}/glove.6B.100d.txt', 'rb') as f:
      for l in f:
          line = l.decode().split()
          word = line[0]
          words.append(word)
          word2idx[word] = idx
          idx += 1
          vect = np.array(line[1:]).astype(float)
          vectors.append(vect)

  zero_embedding = np.zeros(embedding_dim)
  mean_embedding = np.mean(np.array(vectors), axis=0)

  vectors.insert(0,zero_embedding)
  vectors.insert(1,mean_embedding)

  word2idx = {'<pad>': 0, '<unk>': 1, **word2idx}
  words = ["<pad>", "<unk>"] + words

  pickle.dump(vectors, open(f'{glove_path}/6B.100.pkl', 'wb'))
  pickle.dump(words, open(f'{glove_path}/6B.100_words.pkl', 'wb'))
  pickle.dump(word2idx, open(f'{glove_path}/6B.100_idx.pkl', 'wb'))
  print("Vectors and word indexes are created")

vectors = pickle.load(open(f'{glove_path}/6B.100.pkl', 'rb'))
words = pickle.load(open(f'{glove_path}/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.100_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

#Data Preperation
df = pd.read_csv("D:\Gimhan Sandeeptha\Gimhan\Sentiment-Email\LSTM_Production\Datasets\IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

## Data Cleaning

def clean_dataset(df):
    """
    Clean a DataFrame containing a 'review' column.
    Args:
    - df: DataFrame with a 'review' column containing text data.
    Returns:
    - df: DataFrame with the 'review' column replaced by cleaned text.
    """
    cleaned_reviews = []

    for text in df['review']:
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text(" ")

        # Remove non-word characters and digits
        text = re.sub(r"[^\w\s.,]", '', text)
        text = re.sub(r"\d", '', text)

        # Convert to lowercase
        cleaned_text = text.lower()

        cleaned_reviews.append(cleaned_text)

    df['review'] = cleaned_reviews
    return df


cleaned_data = clean_dataset(df)


##Data Preprocessing

X,y = cleaned_data['review'].values,cleaned_data['sentiment'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y)
x_test,x_val,y_test,y_val = train_test_split(x_test,y_test,train_size=0.5,random_state=42,stratify=y_test)

class CustomDataset(Dataset):
    def __init__(self, reviews, sentiments, word2idx, max_seq_length=500,tokenizer = None):
        self.reviews = reviews
        self.sentiments = sentiments
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        label = self.sentiments[index]

        # Tokenize and convert words to indices
        tokens = self.tokenizer(review)
        indices = [self.word2idx.get(token, 0) for token in tokens][:self.max_seq_length]

        # Pad or truncate to a fixed sequence length
        pad_length = self.max_seq_length - len(indices)
        indices += [0] * pad_length  # 0 is used for padding

        # Convert indices to integers
        indices = [int(idx) for idx in indices]

        # One hot encoding
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=2)

        return torch.tensor(indices), one_hot_label.float()

tokenizer = get_tokenizer('basic_english')

# DataLoader for training data
train_dataset = CustomDataset(x_train, y_train, word2idx, max_seq_length=500, tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# DataLoader for validation data
val_dataset = CustomDataset(x_val, y_val, word2idx, max_seq_length=500, tokenizer=tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# DataLoader for testing data
test_dataset = CustomDataset(x_test, y_test, word2idx, max_seq_length=500, tokenizer=tokenizer)
test_loader = DataLoader(test_dataset)


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_layer, hidden_size,
                 num_classes=2,
                 embedding_dim=100,
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.1):
        super(LSTMClassifier, self).__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers = num_layers,
                            bidirectional = bidirectional,
                            dropout=dropout,
                            batch_first=True)

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.sig = nn.Sigmoid()

    def forward(self, input_indices):
        embedded = self.embedding(input_indices)
        fc1_output = self.fc1(embedded)
        lstm_out, _ = self.lstm(fc1_output)
        fc2_output = self.fc2(lstm_out[:, -1, :])
        output = self.sig(fc2_output)
        return output


def create_embedding(weight_dict, trainable=False):
    vocab_size, embedding_dim = len(weight_dict), len(next(iter(weight_dict.values())))

    # Initialize an embedding layer with random weights
    emb_layer = nn.Embedding(vocab_size, embedding_dim)

    # Load the pre-trained weights into the embedding layer
    emb_layer.weight.data.copy_(torch.tensor(list(weight_dict.values())))

    # Set requires_grad based on the trainable flag
    emb_layer.weight.requires_grad = trainable

    return emb_layer

embedding_layer = create_embedding(glove)
weights = embedding_layer.weight.data

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is used")

else:
    device = torch.device('cpu')
    print("GPU not available, CPU used")

hidden_dim = 128
output_dim = 2

# Create the custom model with the embedding layer
model = LSTMClassifier(embedding_layer, hidden_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def match(output,label):
  prediction = int(torch.argmax(output))
  actual_label = int(torch.argmax(label))
  matched = False
  if prediction == actual_label:
    matched = True
  return matched



def train(model,num_epochs,optimizer,loss_function,train_loader,validation_loader,device):
  model = model.to(device)
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      n_correct =0
      n_incorrect =0

      # Training Phase
      for batch_idx, (inputs, labels) in enumerate(train_loader):
          optimizer.zero_grad()
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          loss = loss_function(outputs, labels)
          loss.backward()
          optimizer.step()

          for label,output in zip(labels,outputs):
            if match(output,label):
              n_correct +=1
            else:
              n_incorrect+=1

          total_loss += loss.item()

      average_loss = total_loss / len(train_loader)
      total_accuracy = n_correct/(n_correct+n_incorrect)
      print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss  : {average_loss:.10f}, Training Accuracy  : {total_accuracy:.4f}")


      # Validation phase
      model.eval()
      val_loss = 0
      val_n_correct = 0
      val_n_incorrect = 0

      with torch.no_grad():
          for val_inputs, val_labels in validation_loader:
              val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
              val_outputs = model(val_inputs)
              val_loss += loss_function(val_outputs, val_labels).item()

              for val_label, val_output in zip(val_labels, val_outputs):
                  if match(val_output, val_label):
                      val_n_correct += 1
                  else:
                      val_n_incorrect += 1

      val_average_loss = val_loss / len(validation_loader)
      val_total_accuracy = val_n_correct / (val_n_correct + val_n_incorrect)

      print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_average_loss:.10f}, Validation Accuracy: {val_total_accuracy:.4f}")
      print()

  return model

trained_model = train(model,10,optimizer,criterion,train_loader,val_loader,device)


def test(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Make predictions
            # _, predicted = torch.max(outputs, 1)

            if match(outputs,labels):
              total_correct+=1
            total_samples+=1


    accuracy = total_correct / total_samples
    print(f'Testing Accuracy: {accuracy:.4f}')

test(trained_model,test_loader,device)

model_path = 'D:/Gimhan Sandeeptha/Gimhan/Sentiment-Email/LSTM_Production/Models/lstm_imdb_model.pth'
torch.save(model.state_dict(), model_path)
