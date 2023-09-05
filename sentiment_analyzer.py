# sentiment_analyzer.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    
    def __init__(self, repo_path):
        """
        
        Initializes the SentimentAnalyzer class.
        
        :param repo_path: The path to the downloaded repository containing the model.
        """
        
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(repo_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo_path)
    
    def analyze_text(self, text):
        """
        This function performs sentiment analysis on a single text using the specified model.
        
        :param text: The input text to analyze.
        :return: A tuple containing the predicted label and confidence score.
        """
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Make a prediction using the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted label and confidence score
        predicted_label = outputs.logits.argmax(dim=-1).item()
        confidence_score = outputs.logits.softmax(dim=-1).max().item()
        
        # Convert the predicted label to a string and standardize it
        label_str = self.model.config.id2label[predicted_label]
        
        return label_str, round(confidence_score * 100, 2)
