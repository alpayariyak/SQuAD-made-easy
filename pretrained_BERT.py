import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load Model Fine-Tuned on SQuAD
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'

model = BertForQuestionAnswering.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained(model_name)


# Make a function to predict the answer

def answer_question(question, reference):
    """
    Returns answer to given question by reference
    :param question: Input question
    :param reference: Data to look for answer in
    :return: answer
    """

    # Tokenize question and reference and assign IDs
    token_IDs = tokenizer.encode(question, reference)

    # Locate [SEP] token and split token_IDs into segments corresponding to question and reference
    num_a_tokens = token_IDs.index(tokenizer.sep_token_id) + 1  # +1 to include [SEP] token
    num_b_tokens = len(token_IDs) - num_a_tokens
    mask = [1] * num_a_tokens + [0] * num_b_tokens  # 1 for question, 0 for reference

    if len(mask) != len(token_IDs):
        raise AssertionError('Mask has incorrect length')

    model_output = model(torch.tensor([token_IDs]), token_type_ids=torch.tensor([mask]), return_dict=True)
    start_scores, end_scores = model_output.start_logits, model_output.end_logits

    # Find the combination of start and end tokens that has the highest score
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = tokenizer.decode(token_IDs[answer_start:answer_end + 1])  # +1 to include last token
    return answer
