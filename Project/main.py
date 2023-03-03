from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
import evaluate
from transformers import DataCollatorForSeq2Seq
from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from project_evaluate import compute_metrics

def main():
    # load dataset
    books = load_dataset("opus_books", "en-fr")

    # split dataset to train and test (we don't need it)
    books = books["train"].train_test_split(test_size=0.2)

    # create a tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # create some preprocess function (tokenizer).
    def preprocess_function(examples,  source_lang = "en", target_lang = "fr", prefix = "translate English to French: "):
        inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # tokenize the inputs
    tokenized_books = books.map(preprocess_function, batched=True)

    # create the model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    # data collator adding pads and cut sequence to model maximal length
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # create evaluator metric
    sacrebleu = evaluate.load("sacrebleu")

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="my_awesome_opus_books_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()