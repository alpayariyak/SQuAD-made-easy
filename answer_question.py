import torch
from transformers import BertForQuestionAnswering, BertTokenizer, AutoTokenizer, AutoModelForQuestionAnswering

# Define models
model_name_1 = 'bert-large-uncased-whole-word-masking-finetuned-squad' # BERT model trained on SQuAD
model_name_2 = 'mrm8488/spanbert-finetuned-squadv2' # spanBERT model trained on SQuAD 2.0

models = {
    1: {
        'model': AutoModelForQuestionAnswering.from_pretrained(model_name_1),
        'tokenizer': AutoTokenizer.from_pretrained(model_name_1)
    },
    2: {
        'model': AutoModelForQuestionAnswering.from_pretrained(model_name_2),
        'tokenizer': AutoTokenizer.from_pretrained(model_name_2)
    }
}


# Make a function to predict the answer
def answer_question(question, reference, model_id):
    """
    Returns answer to given question by reference
    :param question: Input question
    :param reference: Data to look for answer in
    :param model_id: ID of model to use for prediction
    :return: answer
    """

    model = models[model_id]['model']
    tokenizer = models[model_id]['tokenizer']

    # Tokenize the question and reference and assign IDs
    token_IDs = tokenizer.encode_plus(question, reference, max_length=512, truncation=True, return_tensors='pt')

    # Extract the tensor containing the token IDs from the dictionary
    input_tokens = token_IDs["input_ids"]

    # Make the model predict the start and end tokens of the answer
    model_output = model(input_tokens)
    start_scores, end_scores = model_output.start_logits, model_output.end_logits

    # Find the combination of start and end tokens that has the highest score
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)+1
    answer = tokenizer.convert_tokens_to_string(token_IDs[answer_start:answer_end])  # +1 to include last token
    return answer

print(answer_question('When did Rollo begin to arrive in Normandy?', 'In the course of the 10th century, the initially destructive incursions of Norse war bands into the rivers of France evolved into more permanent encampments that included local women and personal property. The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, and was situated in the former Frankish kingdom of Neustria. The treaty offered Rollo and his men the French lands between the river Epte and the Atlantic coast in exchange for their protection against further Viking incursions. The area corresponded to the northern part of present-day Upper Normandy down to the river Seine, but the Duchy would eventually extend west beyond the Seine. The territory was roughly equivalent to the old province of Rouen, and reproduced the Roman administrative structure of Gallia Lugdunensis II (part of the former Gallia Lugdunensis)', 2))