from transformers import T5Tokenizer
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM
import os.path
import spacy

from preprocessing import Preprocessor
from project_evaluate import calculate_score
from inference import dependency_postprocessing

def get_generate_comp_tagged(tokenizer, spacy_processor, file_path, write_path, model, generation_arg):
    assert not os.path.exists(write_path), 'Write file already exists'

    preprocessor = Preprocessor(
        '/home/student/Desktop/Project/data/train.labeled',
        '/home/student/Desktop/Project/data/val.labeled',
        tokenizer)

    _, roots, modifiers = preprocessor.clean_text(file_path=file_path, format='unlabeled')
    _, infer_source = preprocessor.read_file(file_path=file_path)

    decoded_preds = list()

    with torch.no_grad():
        for word_input, sentence_roots, sentence_modifiers in tqdm(zip(infer_source, roots, modifiers)):
            input_ids = tokenizer.encode(word_input, return_tensors='pt')
            beam_output = model.generate(
                input_ids,
                max_length=300
                ,num_beams = 50
                , early_stopping=True
            )
            processed_prediction = dependency_postprocessing(tokenizer.decode(beam_output[0], skip_special_tokens=True),
                                                             sentence_roots, sentence_modifiers, spacy_processor)
            decoded_preds.append(processed_prediction)
            # decoded_preds.append(beam_output[0])

    # decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True)

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
                    counter += 1
                    line = f_text[counter]


            if line != '\n':
                f_write.write(line)
            else:
                new_line = f'English:\n{decoded_preds[num_sentences_wrote]}\n\n'
                f_write.write(new_line)
                num_sentences_wrote += 1

            counter += 1

    assert num_sentences_wrote == len(decoded_preds), 'Too many predictions'


def main():
    generate_val = False
    generate_comp = True
    evaluate_val = False
    val_path = 'data/val.unlabeled'
    evaluate_val_path = 'data/val.labeled'
    val_write_path = 'val_337977045_316250877.labeled'
    comp_path = 'data/comp.unlabeled'
    comp_write_path = 'comp_337977045_316250877.labeled'
    model_path = 'models_checkpoints/t5-base-3-19'
    spacy_processor = spacy.load('en_core_web_sm')

    # create a tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    generation_arg = {'num_beams': 50,
                      'early_stopping': True
                      }


    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    print('Generate Validation Predictions:')
    if generate_val:
        get_generate_comp_tagged(tokenizer=tokenizer, spacy_processor=spacy_processor, file_path=val_path, write_path=val_write_path, model=model, generation_arg=generation_arg)

    print('Evaluate Validation Predictions:')
    if evaluate_val:
        print('validation blue score:')
        calculate_score(val_write_path, evaluate_val_path)

    print('Generate Comp Predictions:')
    if generate_comp:
        get_generate_comp_tagged(tokenizer=tokenizer, spacy_processor=spacy_processor, file_path=comp_path, write_path=comp_write_path, model=model, generation_arg=generation_arg)


if __name__ == '__main__':
    main()
