import pandas as pd
import json
import tqdm
import torch
import pickle

from train_unanswerable_classifier import generate_numerical_representations


def answer_question(question, reference, model, tokenizer):
    """
    Returns answer to given question by reference
    :param question: Input question
    :param reference: Input text to look for answer in
    :param model: Model to use for prediction
    :param tokenizer: Tokenizer to use for the model
    :return: answer
    """

    # Tokenize the question and reference and assign IDs
    token_IDs = tokenizer.encode_plus(question, reference, max_length=512, truncation=True, return_tensors='pt')

    # Extract the tensor containing the token IDs from the dictionary
    input_tokens = token_IDs["input_ids"]
    token_type_ids = token_IDs["token_type_ids"]
    attention_mask = token_IDs["attention_mask"]
    # Make the model predict the start and end tokens of the answer
    model_output = model(input_tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
    start_scores, end_scores = model_output.start_logits, model_output.end_logits

    # Find the combination of start and end tokens that has the highest score
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = tokenizer.decode(input_tokens.squeeze()[answer_start:answer_end + 1].tolist())  # +1 to include last token
    special_tokens = ['[SEP]', '[CLS]', '[PAD]', '[UNK]']
    answer = ' '.join([word for word in answer.split() if word not in special_tokens])
    return answer


def is_unanswerable(question, reference):
    """
    Returns whether a question is unanswerable given a reference
    :param question: Input question
    :param reference: Input text to look for answer in
    :return: True if unanswerable, False otherwise
    """
    # Load the trained model from the file
    with open('unanswerable_model.pkl', 'rb') as f:
        unanswerable_classifier = pickle.load(f)

    # Generate numerical representations of the question and reference
    X = generate_numerical_representations(question, reference)
    # Make the model predict whether the question is unanswerable
    y_pred = unanswerable_classifier.predict(X)
    return y_pred[0]


def predict(input_data, model, tokenizer, use_classifier=False):
    """
    Uses a given answer_question function to predict answers and maps them to respective question ID
    :param input_data: pandas DataFrame with 3 columns: question, reference, question_ID
    :param model: model to make predictions with
    :param tokenizer: tokenizer to use for encoding
    :param use_classifier: whether to use the unanswerable classifier
    :return: pandas DataFrame with 2 columns: question_ID, answer
    """

    # Load the trained model from the file
    if use_classifier:
        with open('models/unanswerable_model.pkl', 'rb') as f:
            unanswerable_classifier = pickle.load(f)

    result = []
    for idx, row in tqdm.tqdm(input_data.iterrows(), total=len(input_data), desc='Predicting answers'):
        question = row['question']
        reference = row['reference']
        question_ID = row['question_ID']

        if use_classifier and is_unanswerable(question, reference):
            answer = ''
        else:
            answer = answer_question(question, reference, model, tokenizer)

        answer = answer_question(question, reference, model, tokenizer)
        result.append({'question_ID': question_ID, 'answer': answer})

    return pd.DataFrame.from_records(result)


def prediction_to_json(prediction, path):
    """
    Converts prediction to JSON format
    :param prediction: pandas DataFrame with 2 columns: question_ID, answer
    :param path: path to save JSON to
    :return: None
    """
    result = dict()
    for idx, prediction in prediction.iterrows():
        question_ID = prediction['question_ID']
        answer = prediction['answer']
        result[question_ID] = answer
    with open(path, 'w') as f:
        json.dump(result, f)
