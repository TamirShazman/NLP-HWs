from preprocessing import Preprocessor, max_input_length, max_target_length
import evaluate
from transformers import DataCollatorForSeq2Seq
from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np


def main():
    run_names = 't5-small'
    num_epochs = 50

    # create a tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    #load dataset
    tokenized_dataset =Preprocessor(
                    '/home/student/Desktop/Project/data/train.labeled',
                     '/home/student/Desktop/Project/data/val.labeled',
                     tokenizer).preprocess()

    # create the model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    # config the model
    model.config.max_length = 300
    model.config.early_stopping = True
    model.config.num_beams = 4


    # data collator adding pads and cut sequence to model maximal length
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # create evaluator metric
    sacrebleu = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        #print('\n'+decoded_preds[0])
        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)

        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="runs_outputs",
        evaluation_strategy="epoch",
        run_name=run_names,
        gradient_accumulation_steps=2,
        learning_rate=4e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        save_strategy='epoch',
        #resume_from_checkpoint='/home/student/Desktop/Project/runs_outputs/checkpoint-1248',
        lr_scheduler_type='cosine',
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        logging_strategy='epoch',
        metric_for_best_model='bleu',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )

    trainer.train()
    trainer.save_model(f'/home/student/Desktop/Project/models_checkpoints/{run_names}')

if __name__ == "__main__":
    main()