# app.py

import streamlit as st
import altair as alt
import pandas as pd

from sentiment_analyzer import SentimentAnalyzer

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Define the path to the downloaded repository containing the model
REPO_PATH = "twitter-roberta-base-sentiment-latest"

# Create an instance of the SentimentAnalyzer class
analyzer = SentimentAnalyzer(REPO_PATH)

def benchmark(df):
    """
    This function performs sentiment analysis on all rows of an input DataFrame and returns a DataFrame containing the results.
    
    :param df: The input DataFrame containing the text to analyze.
    :return: A tuple containing a DataFrame with the results of the sentiment analysis and the accuracy of the model.
    """
    
    # Apply the sentiment analysis function to each row of the input DataFrame
    output_df = df.copy()
    output_df[["model_output", "confidence_score"]] = output_df[["text"]].apply(lambda x: pd.Series(analyzer.analyze_text(x["text"])), axis=1)
    
    # Reorder the columns of the output DataFrame
    output_df = output_df[["text", "expected_sentiment", "model_output", "confidence_score"]]
    
    # Calculate the accuracy of the model
    accuracy = (output_df["expected_sentiment"] == output_df["model_output"]).mean()
    
    return output_df, accuracy



def create_chart(label, score):
    """
    This function creates an Altair chart for visualizing the results of sentiment analysis.
    
    :param label: The predicted label from sentiment analysis.
    :param score: The confidence score from sentiment analysis.
    :return: An Altair chart object.
    """
    
    # Create a DataFrame for visualization
    result_df = pd.DataFrame({'label': ['positive', 'negative', 'neutral'], 'score': [0.0, 0.0, 0.0]})
    result_df.loc[result_df['label'] == label, 'score'] = score
    
    # Create an Altair chart object
    c = alt.Chart(result_df.reset_index()).mark_bar().encode(
        x='label',
        y='score',
        color='label',
        text=alt.Text('score', format='.2f')
    ).properties(width=600)
    
    text = c.mark_text(
        align='center',
        baseline='middle',
        dy=-10,
        fontSize=20,
        color='white'
    ).encode(
        text=alt.Text('score', format='.2f')
    )
    
    chart = (c + text).configure_axis(
        labelFontSize=20,
        titleFontSize=20
    )
    
    return chart


# Streamlit app
st.title("Sentiment Analysis")
st.subheader("cardiffnlp/twitter-roberta-base-sentiment-latest")


# Option 1: Enter Text
st.subheader("Option 1: Enter Text")
with st.form(key='nlpForm'):
    raw_text = st.text_area("Enter Text Here")
    submit_button = st.form_submit_button(label='Analyze')

if submit_button:
    label, score = analyzer.analyze_text(raw_text)
    
    # Emoji
    if label == 'positive':
        st.markdown("Sentiment: Positive :smiley:")
    elif label == 'negative':
        st.markdown("Sentiment: Negative :disappointed:")
    else:
        st.markdown("Sentiment: Neutral :neutral_face:")
    
    # Visualization
    chart = create_chart(label, score)
    st.altair_chart(chart, use_container_width=True)

# Option 2: Test Case
st.subheader("Option 2: Test Case")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    test_df = pd.read_csv(uploaded_file)

    index = st.number_input('Enter a row index number', min_value=0, max_value=len(test_df)-1, step=1, value=1)
    if st.button('Display Row'):                
        st.write(test_df.iloc[index])

    if st.button("Benchmark"):
        benchmark_df, accuracy = benchmark(test_df)         
        st.write(f"Accuracy: {accuracy*100:.2f} %")
        st.write(benchmark_df)






