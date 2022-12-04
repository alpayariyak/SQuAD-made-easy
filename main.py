from predict import predict, prediction_to_json
import subprocess
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import argparse
import import_data

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='bert-large-uncased-whole-word-masking-finetuned-squad',
                    help='The name of the model from the hugging face transformers library to use for prediction.')
parser.add_argument('--use_classifier', type=bool, default=False,
                    help='Whether to use the unanswerable classifier or not.')
parser.add_argument('--input_file', type=str, default='data/dev-v2.0.json',
                    help='The path to the input JSON file containing the test data.')
parser.add_argument('--output_file', type=str, default='predictions.json',
                    help='The path to save the output JSON file with the predictions to.')
parser.add_argument('--eval_file', type=str, default='eval_result.json',
                    help='The path to save the JSON file with the evaluation results to.')
args = parser.parse_args()

if __name__ == '__main__':
    # Import the data
    test_data = import_data.import_data(args.input_file, 'test')

    # Load the model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Make predictions
    predictions = predict(test_data, model, tokenizer, args.use_classifier)
    prediction_to_json(predictions, args.output_file)

    # Evaluate the predictions using the official SQuAD evaluation script
    subprocess.run(['python', 'evaluate-v2.0.py', args.input_file, args.output_file,
                    f'--out-file={args.eval_file}'])

