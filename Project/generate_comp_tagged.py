from transformers import T5Tokenizer
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM
import os.path

from preprocessing import Preprocessor
from project_evaluate import calculate_score

def get_generate_comp_tagged(tokenizer, file_path, write_path, model, generation_arg):
    assert not os.path.exists(write_path), 'Write file already exists'

    random.seed(316)

    infer_source, _, _ = Preprocessor(
        '/home/student/Desktop/Project/data/train.labeled',
        '/home/student/Desktop/Project/data/val.labeled',
        tokenizer).clean_text(file_path=file_path, format='unlabeled')

    tokenized_predictions = list()
    with torch.no_grad():
        for word_input in tqdm(infer_source):
            input_ids = tokenizer.encode(word_input, return_tensors='pt')
            beam_output = model.generate(
                input_ids,
                **generation_arg
            )
            tokenized_predictions.append(beam_output[0])

    decoded_preds = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)

    with open(file_path, 'r') as f1:
        f_text = f1.readlines()

    num_lines  = len(f_text)
    num_sentences_wrote = 0
    counter = 0

    with open(write_path, 'w') as f_write:

        while counter < num_lines:

            # last char is \n and we don't need it
            if counter + 1 == num_lines:
                break

            line = f_text[counter]

            if line.startswith('Roots in English:') or line.startswith('Modifiers in English:'):
                while line != '\n':
                    line = f_text[counter]
                    counter += 1

            if line != '\n':
                f_write.write(line)
            else:
                new_line = f'English:\n{decoded_preds[num_sentences_wrote]}\n\n'
                f_write.write(new_line)
                num_sentences_wrote += 1

            counter += 1

    assert num_sentences_wrote == len(decoded_preds), 'To many predictions'


def main():
    generate_val = True
    generate_comp = True
    evaluate_val = True
    val_path = 'data/val.unlabeled'
    val_write_path = 'val_337977045_316250877.labeled'
    comp_path = 'data/comp.unlabeled'
    comp_write_path = 'comp_337977045_316250877.labeled'
    model_path = 'models_checkpoints/t5-small'

    # create a tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    generation_arg = {'max_length': 300,
                      'num_beams': 1,
                      'early_stopping': True
                      }


    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if generate_val:
        get_generate_comp_tagged(tokenizer, val_path, val_write_path, model, generation_arg)

    if generate_comp:
        get_generate_comp_tagged(tokenizer, comp_path, comp_write_path, model, generation_arg)

    if evaluate_val:
        print('validation blue score')
        calculate_score(val_write_path, 'data/val.labeled')

if __name__ == '__main__':
    main()
