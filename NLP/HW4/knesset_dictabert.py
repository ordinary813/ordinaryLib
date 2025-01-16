from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import argparse, os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_masked_sentences",
        type=str,
        help="Path to the masked sentences file."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory for the output files."
    )

    args = parser.parse_args()

    masked_sentences_path = args.path_to_masked_sentences
    out_dir = args.output_dir

    # Create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    # Load the DictaBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert', trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert', trust_remote_code=True)

    model.eval()

    # Open the input file and read its content
    with open(masked_sentences_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        masked_lines = [line.replace('[*]', '[MASK]') for line in lines]

    # Open output file to write the results
    with open(os.path.join(out_dir, 'dictabert_results.txt'), 'w', encoding='utf-8') as output_file:
        # Process each masked sentence
        for line in masked_lines:
            original_sentence = line.strip()

            # Tokenize the sentence
            inputs = tokenizer(original_sentence, return_tensors='pt')

            # Get model predictions
            with torch.no_grad():
                output = model(**inputs)

            # Find all the indexes of the [MASK] tokens
            mask_token_indexes = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

            # List to store all predicted tokens
            predicted_tokens = []

            # Loop through each [MASK] token and predict the top token for that position
            for mask_token_index in mask_token_indexes:
                top_prediction = torch.argmax(output.logits[0, mask_token_index, :]).item()
                predicted_token = tokenizer.convert_ids_to_tokens(top_prediction)
                predicted_tokens.append(predicted_token)

            # Reconstruct the sentence with the generated tokens
            sentence_with_prediction = original_sentence
            for mask_token_index, predicted_token in zip(mask_token_indexes, predicted_tokens):
                sentence_with_prediction = sentence_with_prediction.replace('[MASK]', predicted_token, 1)

            # Write the results to the output file
            output_file.write(f"masked_sentence: {original_sentence}\n")
            output_file.write(f"dictaBERT_sentence: {sentence_with_prediction}\n")
            output_file.write(f"dictaBERT tokens: {', '.join(predicted_tokens)}\n\n")