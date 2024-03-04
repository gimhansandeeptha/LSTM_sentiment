import os
import sys
import zipfile
import pickle
import numpy as np

class GloveEmbedding():
    def __init__(self,folder_path="D:/Gimhan Sandeeptha/Gimhan/Sentiment-Email/LSTM_Production/Datasets/") -> None:
        self.folder_path = folder_path
        self.extracted_folder = "glove"
        self.file_name = "glove.6B.zip"
        self.zip_file_path = os.path.join(self.folder_path, self.file_name)
        self.glove_path = os.path.join(self.folder_path, self.extracted_folder)

    def download_extract_embedding(self):
        if not os.path.exists(self.zip_file_path):
            print("Downloading files...")
            print('''Please fist download the vocab by following power shell command: \n 
                Invoke-WebRequest -Uri "http://nlp.stanford.edu/data/glove.6B.zip" -OutFile {zip_file_path}"''')
            sys.exit()

        if not os.path.exists(os.path.join(self.folder_path, self.extracted_folder)):
            print("Extracting files...")
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.folder_path, self.extracted_folder))

        print("Files are downloaded and extracted successfully")

    def preprocess_embedding(self):

        words = []
        idx = 2
        word2idx = {}
        embedding_dim =100
        vectors = []

        if not os.path.exists(f'{self.glove_path}/6B.100_words.pkl'):
            with open(f'{self.glove_path}/glove.6B.100d.txt', 'rb') as f:
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

            pickle.dump(vectors, open(f'{self.glove_path}/6B.100.pkl', 'wb'))
            pickle.dump(words, open(f'{self.glove_path}/6B.100_words.pkl', 'wb'))
            pickle.dump(word2idx, open(f'{self.glove_path}/6B.100_idx.pkl', 'wb'))
            print("Vectors and word indexes are created")

    def get_embedding(self):
        vectors = pickle.load(open(f'{self.glove_path}/6B.100.pkl', 'rb'))
        words = pickle.load(open(f'{self.glove_path}/6B.100_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{self.glove_path}/6B.100_idx.pkl', 'rb'))

        glove = {w: vectors[word2idx[w]] for w in words}
        return vectors, words, word2idx, glove

