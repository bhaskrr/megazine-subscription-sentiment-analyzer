import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

class PreprocessAndVectorize:
    def __init__(self, glove_file_path, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.embeddings_index = self.load_glove_embeddings(glove_file_path)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def load_glove_embeddings(self, file_path):
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
        return embeddings_index
    
    def remove_stopwords(self, text):
        filtered_words = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    
    def process(self, text):
        text = text.lower().strip()

        # Remove URLs and unwanted characters
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
        text = re.sub(r'[\(\)\[\]]', '', text)
        text = re.compile('<.*?>').sub('', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Uncomment this line if you want to remove stopwords
        # text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        
        return text
    
    def text_to_glove_vector(self, text):
        words = text.split()
        word_vectors = [self.embeddings_index.get(word, np.zeros(self.embedding_dim)) for word in words]

        # Return the average of the word vectors (reshape to (1, 100))
        return np.mean(word_vectors, axis=0).reshape(1, -1) if word_vectors else np.zeros((1, self.embedding_dim))
    
    def dataset_to_glove(self, X):
        return np.array([self.text_to_glove_vector(text) for text in X])