import pandas as pd
import json
import tqdm


def import_data(path, type, fraction=1):
    """
    Create pandas DataFrame with 3 columns: question, reference, answer
    :param path: path to JSON
    :param type: 'train' or 'test'
    :param fraction: fraction of data to import
    :return: pandas DataFrame
    """
    result = []
    with open(path, 'r') as f:
        data = json.load(f)
        for full_text in data['data']:
            for paragraph in full_text['paragraphs']:
                reference = paragraph['context']
                for question_answer in paragraph['qas']:
                    question = question_answer['question']
                    if type == 'train':
                        answer = question_answer['answers'][0]['text'] if not question_answer['is_impossible'] else ''
                        result.append({'question': question, 'reference': reference, 'answer': answer})
                    else:
                        question_ID = question_answer['id']
                        result.append({'question': question, 'reference': reference, 'question_ID': question_ID})

    result = pd.DataFrame.from_records(result)

    if fraction != 1:  # if fraction is not 1, take a random sample of the data
        result = result.sample(frac=fraction)

    return result


def predict(input_data, answer_question_function):
    """
    Uses a given answer_question function to predict answers and maps them to respective question ID
    :param input_data: pandas DataFrame with 3 columns: question, reference, question_ID
    :param answer_question_function: function that takes question and reference as input and returns answer
    :return: pandas DataFrame with 2 columns: question_ID, answer
    """
    result = []
    for idx, row in tqdm.tqdm(input_data.iterrows(), total=len(input_data), desc='Predicting answers'):
        question = row['question']
        reference = row['reference']
        question_ID = row['question_ID']
        answer = answer_question_function(question, reference)
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