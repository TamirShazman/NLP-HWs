import os
import pandas as pd
import string
import re
from datasets import Dataset
from transformers import AutoTokenizer

max_input_length = 128
max_target_length = 128


class Preprocessor:
    def __init__(self, train_path, validation_path, tokenizer):
        self.default_ip = '!?.'
        self.default_st_tagging = {"German:\n": "SOURCE ", "English:\n": "SPLIT "}
        self.default_rm_tagging = {"Roots in English: ": "SPLIT ", "Modifiers in English: ": "SPLIT "}
        self.train_path = train_path
        self.validation_path = validation_path
        self.tokenizer = tokenizer


    def clean_text(self, file_path, format="labeled"):
        """
        Assumes strict order of German than English
        Returns two lists, source and target each of which is made up of multiple sentences. (not a list of words). Should
        be better for hugging face interface.
        @param file_path: path to file holding text
        @param format: allows function to cover both types of file inputs. "labeled" returns source and target lists, while
        "unlabeled" returns source + root lists and modifiers list of tuples
        @param ignored_punctuation: list of what punctuation to leave in sentence
        @param source_target_tagging: dictionary for tagging what part of text is target and source
        @param root_modifier_tagging: dictionary for tagging what part of text is root and modifiers
        @return: depending on label either source + target lists or source + roots + modifier lists
        """
        ignored_punctuation = self.default_ip
        source_target_tagging = self.default_st_tagging
        root_modifier_tagging = self.default_rm_tagging

        # read file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # initial cleaning

        pattern = '|'.join(sorted(re.escape(obj) for obj in source_target_tagging))
        tagged_text = re.sub(pattern, lambda m: source_target_tagging.get(m.group(0)), text, flags=re.IGNORECASE)
        if format == "unlabeled":
            pattern2 = '|'.join(sorted(re.escape(obj) for obj in root_modifier_tagging))
            tagged_text = re.sub(pattern2, lambda m: root_modifier_tagging.get(m.group(0)), tagged_text,
                                 flags=re.IGNORECASE)
            ignored_punctuation += '()'
            # ignored_punctuation = ignored_punctuation + '()'
        regex_cleaning = dict()
        regex_cleaning.update({'\n': ' '})
        regex_cleaning.update({p: '' for p in string.punctuation if p not in ignored_punctuation})
        clean_text = tagged_text.translate(str.maketrans(regex_cleaning))
        # clean_text = tagged_text
        action_items = clean_text.split("SOURCE")

        # reorganization
        if format == "labeled":
            source_list = list()
            target_list = list()

            for action_item in action_items[1:]:
                source_target_obj = action_item.split("SPLIT")
                source_target_obj = [st_text.strip() for st_text in source_target_obj]
                source_list.append("translate German to English: " + source_target_obj[0])
                target_list.append(source_target_obj[1])
            return source_list, target_list
        elif format == "unlabeled":
            source_list = list()
            root_list = list()
            modifier_list = list()
            for action_index, action_item in enumerate(action_items[1:]):
                source_root_modifier_obj = action_item.split("SPLIT")
                source_root_modifier_obj = [st_text.strip() for st_text in source_root_modifier_obj]
                source_list.append("translate German to English: " + source_root_modifier_obj[0].translate(
                    str.maketrans({"(": "", ")": ""})))
                root_list.append(source_root_modifier_obj[1].split(' '))
                modifier_tuple_list = list()
                modifier_tuples = source_root_modifier_obj[2].translate(str.maketrans({"(": "*", ")": "*"})).split(
                    "*")
                for tup_index in range(0, len(modifier_tuples) - 1, 2):
                    modifier_tuple_list.append(tuple(modifier_tuples[tup_index + 1].split(' ')))
                modifier_list.append(modifier_tuple_list)
            return source_list, root_list, modifier_list
        else:
            raise ("Error: no process completed")

    def check_file_statistics(self, filepath):
        """
        Receives file pathway, loads and cleans data. Then prints statistics about the length of sentences, or root/modifier input
        @param filepath:
        @return:
        """
        format = filepath.split('.')[-1]
        if format == "labeled":
            source_list, target_list = self.clean_text(filepath, format=format)
            interest_dict = {"source_length": list(), "target_length": list()}
            for s, t in zip(source_list, target_list):
                interest_dict["source_length"].append(len(s.split(' ')))
                interest_dict["target_length"].append(len(t.split(' ')))

        elif format == "unlabeled":
            source_list, root_list, modifier_list = self.clean_text(filepath, format=format)
            interest_dict = {"source_length": list(), "root_length": list(), "modifier_length": list()}
            for s, r, m in zip(source_list, root_list, modifier_list):
                interest_dict["source_length"].append(len(s.split(' ')))
                interest_dict["root_length"].append(len(r))
                interest_dict["modifier_length"].append(len(m))
        else:
            raise ("Does not conform to either either labeled or unlabeled format")
        print(pd.DataFrame.from_dict(interest_dict).describe())

    def preprocess(self):
        train_source_list, train_target_list = self.clean_text(self.train_path)
        train_pre_ds = [{'de': german, 'en': english} for german, english in zip(train_source_list, train_target_list)]
        validation_source_list, validation_target_list = self.clean_text(self.validation_path)
        validation_pre_ds = [{'de': german, 'en': english} for german, english in zip(validation_source_list, validation_target_list)]

        train_ds = Dataset.from_pandas(pd.DataFrame({'id': list(range(len(train_source_list))), 'translation': train_pre_ds}))
        validation_ds = Dataset.from_pandas(pd.DataFrame({'id': list(range(len(validation_source_list))), 'translation': validation_pre_ds}))

        tokenizer = self.tokenizer
        # %%


        def preprocess_function(dataset):
            source = [text['de'] for text in dataset['translation']]
            target = [label['en'] for label in dataset['translation']]
            model_inputs = tokenizer(source, return_tensors='np', max_length=max_input_length, truncation=True, padding=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(target, return_tensors='np', max_length=max_target_length, truncation=True, padding=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = train_ds.map(preprocess_function, batched=True) #.remove_columns(['text', 'label'])

        tokenized_validation = validation_ds.map(preprocess_function, batched=True)

        return {"train": tokenized_train, "validation": tokenized_validation}