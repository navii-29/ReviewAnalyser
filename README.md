# IMDB Movie Review Sentiment Analysis

A web application that predicts the sentiment of movie reviews using a deep learning model built with TensorFlow/Keras and deployed with Streamlit.

The model analyzes text reviews and classifies them as **Positive** or **Negative** based on learned patterns from the IMDB movie review dataset.

## Overview

This project demonstrates how Natural Language Processing (NLP) and Recurrent Neural Networks (RNNs) can be used for sentiment analysis.
A trained GRU-based model processes user input, converts the text into numerical sequences, and predicts the sentiment score.

The interface allows users to enter a movie review and instantly receive a sentiment prediction.

## Features

* Sentiment prediction for movie reviews
* GRU-based deep learning model
* Text preprocessing and tokenization
* Out-of-vocabulary word handling
* Streamlit-based web interface
* Confidence score visualization

## Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Regular Expressions

## Dataset

The model uses the **IMDB Movie Reviews Dataset**, a widely used dataset for binary sentiment classification.

* 50,000 movie reviews
* Balanced positive and negative samples
* Pre-tokenized dataset provided by Keras

## Project Structure

```
IMDB-Sentiment-Analyzer
│
├── app.py                     # Streamlit application
├── GRU_sentiment_model.h5     # Trained deep learning model
├── requirements.txt           # Project dependencies
└── README.md
```

## How It Works

1. User enters a movie review in the web interface.
2. The text is cleaned and normalized.
3. Words are converted into numerical indices using the IMDB word index.
4. The sequence is padded to a fixed length.
5. The trained GRU model predicts the sentiment score.
6. The application displays the predicted sentiment and confidence score.

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

Install dependencies:

```
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit server:

```
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Example

Input:

```
The movie was amazing and the acting was brilliant.
```

Output:

```
Sentiment: Positive
Confidence Score: 0.92
```

## Model Details

* Architecture: GRU-based Recurrent Neural Network
* Embedding layer for word representations
* Binary classification using sigmoid activation
* Sequence padding length: 500 tokens

## Future Improvements

* Use transformer-based models (BERT / DistilBERT)
* Add batch predictions
* Deploy the application online
* Add more dataset support

## License

This project is open-source and available under the MIT License.
