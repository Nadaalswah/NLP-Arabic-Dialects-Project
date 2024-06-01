from flask import Flask, request, render_template, jsonify
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import emoji
import nltk

# Download the 'punkt' tokenizer if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define the ArabicTextCleaner class
class ArabicTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, use_stemming=False):
        self.use_stemming = use_stemming
        self.arabic_stopwords = set(stopwords.words('arabic'))
        
        if self.use_stemming:
            self.stemmer = stemmer("arabic")
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, str):
            X = [X]
        return [self.clean_text(text) for text in X]
    
    def clean_text(self, text):
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        text = text.replace('ـ', '')
        text = re.sub(r'<.*?>', '', text)
        words = word_tokenize(text)
        text = re.sub(r'[0-9’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', '', ' '.join(words))
        if self.use_stemming:
            words = [self.stemmer.stemWord(word) for word in words]
        text = ' '.join(words)
        text = re.sub(r'#([^\s]+)', '', text)
        text = re.sub(r'http\S+|www\S+|@\S+', '', text)
        text = ''.join(char for char in text if not emoji.is_emoji(char))
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        return text

app = Flask(__name__)

# Load the preprocessing pipeline, vectorizer, and classifier using raw strings
preprocessor = joblib.load(r'NLP\arabic_text_cleaner_pipeline.pkl')
vectorizer = joblib.load(r'NLP\CountVectorizer_test.joblib')
classifier = joblib.load(r'NLP\model\CountVectorizer_model.pkl')

# Define the label mapping
label_mapping = {0: 'EG', 1: 'LB', 2: 'LY', 3: 'MA', 4: 'SD'}

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('logo.html')

@app.route('/take_photo', methods=['GET'])
def take_photo():
    return render_template('photo.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        if not text:
            return render_template('photo.html', prediction_text="No text provided.")
        
        # Preprocess the text
        cleaned_text = preprocessor.transform([text])
        
        # Vectorize the cleaned text
        vectorized_text = vectorizer.transform([text])
        
        # Predict using the classifier
        prediction = classifier.predict(vectorized_text)
        
        # Map the prediction to the corresponding label
        predicted_label = label_mapping[prediction[0]]
        
        return render_template('photo.html', prediction_text=f'Predicted Dialect: {predicted_label}')

if __name__ == '__main__':
    app.run(debug=True)
