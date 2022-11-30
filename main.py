import answer_question
from predict import import_data, predict, prediction_to_json
import subprocess

if __name__ == '__main__':
    # test_data = import_data('data/dev-v2.0.json', 'test')  # import test data
    # predictions = predict(test_data, pretrained_BERT.answer_question)  # predict answers
    # prediction_to_json(predictions, 'predictions.json') # save predictions to JSON
    subprocess.run(['python', 'evaluate-v2.0.py', 'data/dev-v2.0.json', 'predictions.json', '--out-file=eval.json'])  # evaluate predictions

