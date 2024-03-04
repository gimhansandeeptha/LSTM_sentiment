import torch.nn as nn

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
    