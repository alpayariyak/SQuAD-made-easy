from answer_question import answer_question
from predict import import_data, predict, prediction_to_json
import subprocess

models_to_test = [('BERT_large_finetuned', 1), ('spanBERT', 2)]

if __name__ == '__main__':
    for model_name, model_id in models_to_test:
        test_data = import_data('data/dev-v2.0.json', 'test')  # import test data
        predictions = predict(test_data, answer_question, model_id)  # predict answers
        prediction_to_json(predictions, f'predictions_{model_name}.json')  # save predictions to JSON
        subprocess.run(['python', 'evaluate-v2.0.py', 'data/dev-v2.0.json', f'predictions_{model_name}.json', f'--out-file=eval_{model_name}.json'])  # evaluate predictions

