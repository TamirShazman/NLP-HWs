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

def make_sure_mod_root(file_path, new_path):
    with open(file_path, 'r') as file:
        # read a list of lines into data
        lines = file.readlines()

    new_lines = []
    skip_till_new_line = False
    for line in lines:

        if line == '\n':
            skip_till_new_line = False

        if skip_till_new_line:
            continue

        cut_from = None
        start_index_root = None
        start_index_mod = None

        if 'roots in' in line.lower():
            start_index_root = line.lower().index('roots in')
        if 'lost modifiers are:' in line.lower():
            start_index_mod = line.lower().index('lost modifiers are:')
        if 'modifiers in' in line.lower():
            start_index_mod = line.lower().index('modifiers in')
        if 'modifier in' in line.lower():
            start_index_mod = line.lower().index('modifier in')
        if 'modifiers' in line.lower():
            start_index_mod = line.lower().index('modifiers')


        if start_index_root is None and start_index_mod is not None:
            cut_from = start_index_mod
        elif start_index_root is not None and start_index_mod is None:
            cut_from = start_index_root
        elif start_index_root is not None and start_index_mod is not None:
            cut_from = min(start_index_root, start_index_mod)

        if cut_from is None:
            new_lines.append(line)
        else:
            new_lines.append(line[:cut_from].strip())
            skip_till_new_line = True

    with open(new_path, 'w') as f:
        for line in new_lines:
            if not line.endswith('\n'):
                f.write(f"{line}\n")
            else:
                f.write(line)

        f.write('\n')


def get_generate_comp_tagged(tokenizer, spacy_processor, file_path, write_path, model, generation_arg):
    """
    Create predictions and write to write path
    :param tokenizer:
    :param spacy_processor:
    :param file_path:
    :param write_path:
    :param model:
    :param generation_arg:
    :return:
    """
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
    generate_val = True
    generate_comp = True
    evaluate_val = True
    val_path = 'data/val.unlabeled'
    evaluate_val_path = 'data/val.labeled'
    temp_val_path = 'tests/val_temp.labeled'
    val_write_path = 'val_337977045_316250877.labeled'
    comp_path = 'data/comp.unlabeled'
    temp_comp_path = 'tests/comp_temp.labeled'
    comp_write_path = 'comp_337977045_316250877.labeled'
    model_path = 'models_checkpoints/t5-base-3-19'
    spacy_processor = spacy.load('en_core_web_sm')

    # create a tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    generation_arg = {'num_beams': 50,
                      'early_stopping': True
                      }


    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


    if generate_val:
        print('Generate Validation Predictions:')
        get_generate_comp_tagged(tokenizer=tokenizer, spacy_processor=spacy_processor, file_path=val_path, write_path=temp_val_path, model=model, generation_arg=generation_arg)
        make_sure_mod_root(temp_val_path, val_write_path)


    if evaluate_val:
        print('Evaluate Validation Predictions:')
        print('validation blue score:')
        calculate_score(val_write_path, evaluate_val_path)


    if generate_comp:
        print('Generate Comp Predictions:')
        get_generate_comp_tagged(tokenizer=tokenizer, spacy_processor=spacy_processor, file_path=comp_path, write_path=temp_comp_path, model=model, generation_arg=generation_arg)
        make_sure_mod_root(temp_comp_path, comp_write_path)

if __name__ == '__main__':
    main()
