from xgboost import XGBClassifier
import import_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
import re
import nltk
import pickle


def clean_text(text):
    """
    Clean and normalize the input text
    :param text: input text
    :return: cleaned and normalized text
    """
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # Remove numbers and whitespace
    text = re.sub(r'\d+', '', text)
    text = text.strip()

    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    text = ' '.join(filtered_tokens)

    return text


def generate_numerical_representations(questions, references):
    """
    Generate numerical representations of the input text using an NLP model
    :param questions: list of questions
    :param references: list of corresponding references
    :return: numerical representations of the input text
    """
    # Combine the questions and references into a single list
    texts = questions + references

    # Use a CountVectorizer to generate numerical representations of the text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    return X


def train(X, y):
    """
    Train a binary classification model
    :param X: input features
    :param y: labels
    :return: trained model
    """
    # Use XGBoost(from my love for Kaggle) for binary classification
    model = XGBClassifier(use_label_encoder=False)
    model.fit(X, y)

    return model


def evaluate(model, X, y):
    """
    Evaluate the performance of a binary classification model
    :param model: trained model
    :param X: input features
    :param y: labels
    :return: dictionary of evaluation metrics
    """
    # Make predictions using the trained model
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Return the evaluation metrics as a dictionary
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == '__main__':
    # Import the data
    data = import_data.import_data('data/train-v2.0.json', 'train')
    # Clean and normalize the questions and references
    data['question'] = data['question'].apply(clean_text)
    data['reference'] = data['reference'].apply(clean_text)

    # Generate numerical representations of the questions and references
    X = generate_numerical_representations(data['question'], data['reference'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, data['is_impossible'], test_size=0.2)

    # Train a binary classification model
    model = train(X_train, y_train)

    # Evaluate the performance of the model
    metrics = evaluate(model, X_test, y_test)
    print(metrics)

    # Save the trained model to a file
    with open('models/unanswerable_model.pkl', 'wb') as f:
        pickle.dump(model, f)