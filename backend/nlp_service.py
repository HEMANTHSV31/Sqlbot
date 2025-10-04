import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from typing import Dict,Optional

class NlpService:
    """
    A service to determine user intent and extract entities from text.
    This is a simple, effective model for routing user requests.
    """
    def __init__(self, data_file: str = "cmd_data.csv"):
        self.data_file = data_file
        self.model: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.load_data()

    def load_data(self):
        """Loads training data and trains the intent classification model."""
        try:
            df = pd.read_csv(self.data_file)
        except FileNotFoundError:
            raise RuntimeError(f"NLP data file not found: {self.data_file}. Please create it.")

        # Simple cleaning to improve matching
        df['text'] = df['text'].str.lower().str.strip()
        
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(df['text'])
        self.model = LogisticRegression()
        self.model.fit(X, df['intent'])
        print(f"NLP Model trained on {len(df)} examples.")

    def predict(self, text: str) -> Dict:
        """
        Predicts the intent and extracts entities from the user's text.
        """
        text_clean = text.lower().strip()
        
        # Predict the intent (e.g., 'get_data', 'greeting', 'unknown')
        X_pred = self.vectorizer.transform([text_clean])
        intent = self.model.predict(X_pred)[0]
        
        # Simple entity extraction (can be expanded)
        entities = {}
        
        # Look for numbers, which could be IDs or limits
        numbers = re.findall(r'\d+', text)
        if numbers:
            entities['limit'] = int(numbers[0]) # Assume first number is a limit
            entities['number'] = int(numbers[0])

        return {
            "intent": intent,
            "entities": entities,
            "original_text": text
        }