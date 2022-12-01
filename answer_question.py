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
