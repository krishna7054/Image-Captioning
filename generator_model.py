import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class GeneratorModel:
    def __init__(self, image_path, model_path, tokenizer_path, feature_path, max_length):
        self.image_path = image_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.feature_path = feature_path
        self.max_length = max_length
        
    def idx_to_word(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
        
    def predict_caption(self, model, image_path, tokenizer, max_length, features):
        print(image_path)
        feature = features[image_path]
        feature = np.squeeze(feature, axis=0)
        in_text = 'startseq'
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # Ensure sequence length matches max_length
            sequence = pad_sequences([sequence], maxlen=max_length)
            y_pred = model.predict([np.expand_dims(feature, axis=0), sequence], verbose=0)
            y_pred = np.argmax(y_pred)
            word = self.idx_to_word(y_pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text
    
    def  generate_caption(self):
        caption_model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
            handle.close()
        with open(self.feature_path, 'rb') as handle:
            features = pickle.load(handle)
            handle.close()
            caption = self.predict_caption(caption_model, self.image_path, tokenizer, self.max_length, features)
            caption = caption[len('startseq'):caption.rfind('endseq')].strip()
        return f'Generated Caption: {caption}'