# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import re
import numpy as np
from keras.utils import pad_sequences

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('GRU_sentiment_model.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input

 


def preprocess_text(text, word_index, max_len, oov_index=2):
    # Lowercase
    text = text.lower()

    # Normalize quotes ( “ ” ‘ ’ -> " ')
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)

    # Remove punctuation but keep words
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Encode (true OOV handling)
    encoded = [word_index[word] + 3 if word in word_index else oov_index for word in words]

    # Pad
    padded = pad_sequences(
        [encoded],
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    return padded.astype("int32")




import streamlit as st
import numpy as np

# Page config
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f9f9f9;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.result-positive {
    color: #0f9d58;
    font-size: 28px;
    font-weight: bold;
}
.result-negative {
    color: #db4437;
    font-size: 28px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🎬 IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning | Simple RNN | NLP</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# Main card
st.markdown('<div class="card">', unsafe_allow_html=True)

st.write("### ✍️ Enter a movie review")
user_input = st.text_area(
    "",
    placeholder="Type your movie review here...",
    height=200
)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        preprocessed_input = preprocess_text(user_input,word_index,500)
        prediction = model.predict(preprocessed_input)

        score = float(prediction[0][0])
        sentiment = "Positive 😄" if score > 0.5 else "Negative 😞"

        st.write("---")
        st.write("### 📊 Prediction Result")

        if score > 0.5:
            st.markdown(f'<div class="result-positive">{sentiment}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-negative">{sentiment}</div>', unsafe_allow_html=True)

        st.progress(score)
        st.write(f"**Confidence Score:** `{score:.2f}`")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
🚀 Built with TensorFlow, Keras & Streamlit<br>
📌 Dataset: IMDB Reviews
</div>
""", unsafe_allow_html=True)
