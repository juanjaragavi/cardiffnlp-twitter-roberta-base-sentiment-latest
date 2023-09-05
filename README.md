# Sentiment Analysis App

* Model is now hosted from streamlit cloud [twitter-roberta-base-sentiment-latest](https://cardiffnlp-twitter-roberta-base-sentiment-latest-rcruzin-ai.streamlit.app)
* This is a sentiment analysis app that uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model from HuggingFace. 
* The app is built using Streamlit and allows users to perform sentiment analysis on text data and test cases in csv file.


## Model

The sentiment analysis model used in this app is `cardiffnlp/twitter-roberta-base-sentiment-latest`. You can find more information about the model on its [HuggingFace page](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

## Installation

To install the required libraries for this app, run the following command in your terminal:

```
pip install -r requirements.txt
```

This will install all the libraries listed in the `requirements.txt` file, including `streamlit`, `altair`, `transformers`, and others.

## Usage

To run the app, navigate to the project directory in your terminal and run the following command:

```
streamlit run app.py
```

This will start the Streamlit server and open the app in your web browser. You can then use the app to perform sentiment analysis on text data.

## Files

- `sentiment_analyzer.py`: This file contains the `SentimentAnalyzer` class, which is used to perform sentiment analysis on text data using the specified model.
- `app.py`: This file contains the Streamlit app code, which provides a user interface for performing sentiment analysis using the `SentimentAnalyzer` class.
