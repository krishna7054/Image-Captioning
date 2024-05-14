# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model

# # Load the tokenizer
# with open('tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

# # Load the features dictionary
# with open('features.pkl', 'rb') as f:
#     features = pickle.load(f)

# # Load the trained captioning model
# caption_model = load_model('model.h5')

# # Function to generate captions for a given image
# def generate_caption(image_path, model, tokenizer, max_length, features):
#     feature = features[image_path]
#     in_text = 'startseq'
#     for _ in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         y_pred = model.predict([feature, sequence], verbose=0)
#         y_pred = np.argmax(y_pred)
#         word = idx_to_word(y_pred, tokenizer)
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word == 'endseq':
#             break
#     return in_text

# # Function to convert an integer sequence to a word
# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None

# # Provide the path of the image you want to test
# image_path_to_test = 'image-6.jpg'  # Update this with your image path

# # Generate and print the caption for the image
# caption = generate_caption(image_path_to_test, caption_model, tokenizer, max_length, features)
# print('Generated Caption:', caption)


import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to convert an integer sequence to a word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to extract features from a new image
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Rescale pixel values to [0, 1]

    # Load the same DenseNet201 model used during training
    model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

    features = model.predict(image)
    return features

# Function to generate captions for a given image
def generate_caption(image_features, model, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([image_features, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the trained captioning model
caption_model = load_model('model.h5')

def process_image_caption(image_path):
    # Provide the path of the new image you want to test
    image_path_to_test = image_path

    # Extract features from the new image
    image_features = extract_features(image_path_to_test)

    # Generate the caption for the new image
    caption = generate_caption(image_features, caption_model, tokenizer, max_length=34)  # Assuming max_length is 34
    caption = caption[len('startseq'):caption.rfind('endseq')].strip()
    return caption
