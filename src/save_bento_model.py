from pathlib import Path
import torch
import bentoml

model_path = "D:\Gimhan Sandeeptha\Gimhan\Sentiment-Email\LSTM_Production\Models\lstm_imdb_model.pth"
def load_and_save_model(model_path) -> None:
    model = torch.load(model_path)
    bentoml.pytorch.save_model("lstm_model",
                                model,
                                signatures={"__call__":{"batchable":True, "batch_dim":0}}
                                )
if __name__ == "__main__":
    load_and_save_model(Path(model_path))