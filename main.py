from answer_question import answer_question
from predict import import_data, predict, prediction_to_json
import subprocess

if __name__ == '__main__':
    # test_data = import_data('data/dev-v2.0.json', 'test')  # import test data
    # predictions = predict(test_data, answer_question, 2)  # predict answers
    # prediction_to_json(predictions, 'predictions_spanBERT.json') # save predictions to JSON
    subprocess.run(['python', 'evaluate-v2.0.py', 'data/dev-v2.0.json', 'predictions_spanBERT.json', '--out-file=eval_spanBERT.json'])  # evaluate predictions

