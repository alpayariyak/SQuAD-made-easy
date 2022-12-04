import json
import pandas as pd


def import_data(path, data_type, fraction=1):
    """
    Create pandas DataFrame with 3 columns: question, reference, answer
    :param path: path to JSON
    :param data_type: 'train' or 'test'
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
                    if data_type == 'train':
                        answer = question_answer['answers'][0]['text'] if not question_answer['is_impossible'] else ''
                        result.append({'question': question, 'reference': reference, 'answer': answer,
                                       'is_impossible': int(question_answer['is_impossible'])})
                    else:
                        question_ID = question_answer['id']
                        result.append({'question': question, 'reference': reference, 'question_ID': question_ID})

    return pd.DataFrame.from_records(result).sample(frac=fraction)