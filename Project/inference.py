from preprocessing import Preprocessor, max_input_length, max_target_length
import evaluate
from transformers import DataCollatorForSeq2Seq
from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline
import numpy as np
import pickle
from time import time
import random
import torch
from tqdm import tqdm

def dependency_postprocessing(sentences, roots, modifiers, spacy_processor):
    """
    Identify sentence root, replace if necessary with given roots.
    Identify type of speech of modifiers, identify children of root with that type
    of speech and replace with modifiers
    :param sentence:
    :param roots:
    :param modifiers:
    :param spacy_processor:
    :return:
    """
    # get object with dependencies and POS of sentence
    try:
        spacy_doc = spacy_processor(sentences)
        roots = [sent.root for sent in spacy_doc.sents]
        childrens = [[child for child in sent.root.children] for sent in spacy_doc.sents]

        final_sentence = ''
        # for each sentence check if root appears multiple times, replace only root with given root.
        # Then replace children of root of the same POS with modifers
        for sentence, root, children in zip(spacy_doc.sents, roots, childrens):
            if sentence.text.count(root) > 1:
                i = 1
                for token in sentence:
                    if token.text == root and token.dep_ != "ROOT":
                        i = i + 1
                    elif token.dep_ == "ROOT":
                        break
                    else:
                        continue



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    decoded_preds, labels = postprocess_text(decoded_preds, labels)

    # print('\n'+decoded_preds[0])
    result = metric.compute(predictions=decoded_preds, references=labels)

    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["avg_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def main():
    run_names = 't5-small'
    num_epochs = 50

    # create a tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    sacrebleu = evaluate.load("sacrebleu")

    # #load dataset
    infer_source, infer_target =Preprocessor(
                    '/home/student/Desktop/Project/data/train.labeled',
                     '/home/student/Desktop/Project/data/val.labeled',
                     tokenizer).clean_text('/home/student/Desktop/Project/data/val.labeled')
    infer_source, roots, modifiers = Preprocessor(
                    '/home/student/Desktop/Project/data/train.labeled',
                     '/home/student/Desktop/Project/data/val.labeled',
                     tokenizer).clean_text('/home/student/Desktop/Project/data/val.unlabeled', format='unlabeled')
    # # sampling
    random.seed(316)
    random_sampling_indexes = random.sample(range(0, len(infer_source)), 5)
    infer_source = [infer_source[i] for i in random_sampling_indexes]
    infer_target = [infer_target[i] for i in random_sampling_indexes]
    print(len(infer_source), len(infer_target))

    for model_name in ['t5-base']:
        # create the model
        model = AutoModelForSeq2SeqLM.from_pretrained("/home/student/Desktop/Project/models_checkpoints/" + model_name)
        for beam_size in [50]:
            print(model_name)
            print(beam_size)
            now = time()
            tokenized_predictions = list()
            with torch.no_grad():
                for word_input, root in tqdm(zip(infer_source, roots)):
                    input_ids = tokenizer.encode(word_input, return_tensors='pt')

                    force_words_ids = tokenizer(root, add_special_tokens=False).input_ids
                    # constraints = []
                    # for word in root:
                    #     constraints.append(PhrasalConstraint(tokenizer(word).input_ids))

                    beam_output = model.generate(
                        input_ids,
                        max_length=int(len(word_input) * 1.1),
                        # force_words_ids=force_words_ids,
                        num_beams=beam_size,
                        early_stopping=True
                    )
                    tokenized_predictions.append(beam_output[0])
            total_time = time() - now
            bleu_score = compute_metrics((tokenized_predictions, infer_target), tokenizer, sacrebleu)

            print(total_time, bleu_score)

        # print("Output:\n" + 100 * '-')
        # print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
    #
    # # config the model
    # model.config.max_length = 300
    # model.config.early_stopping = True
    # model.config.num_beams = 4
    #
    #
    # # data collator adding pads and cut sequence to model maximal length
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    #
    # # create evaluator metric
    # sacrebleu = evaluate.load("sacrebleu")
    #
    # def postprocess_text(preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [[label.strip()] for label in labels]
    #
    #     return preds, labels
    #
    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    #     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    #     #print('\n'+decoded_preds[0])
    #     result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    #
    #     result = {"bleu": result["score"]}
    #
    #     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #     result["gen_len"] = np.mean(prediction_lens)
    #     result = {k: round(v, 4) for k, v in result.items()}
    #     return result
    #
    # # training args
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="runs_outputs",
    #     evaluation_strategy="epoch",
    #     run_name=run_names,
    #     gradient_accumulation_steps=2,
    #     learning_rate=4e-3,
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     weight_decay=0.01,
    #     save_total_limit=3,
    #     num_train_epochs=num_epochs,
    #     save_strategy='epoch',
    #     #resume_from_checkpoint='/home/student/Desktop/Project/runs_outputs/checkpoint-1248',
    #     lr_scheduler_type='cosine',
    #     predict_with_generate=True,
    #     fp16=True,
    #     load_best_model_at_end=True,
    #     logging_strategy='epoch',
    #     metric_for_best_model='bleu',
    # )
    #
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["validation"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    #
    # )
    #
    # trainer.train()
    # trainer.save_model(f'/home/student/Desktop/Project/models_checkpoints/{run_names}')

if __name__ == "__main__":
    main()