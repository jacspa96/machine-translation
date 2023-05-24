from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
import string
import re

SOS = "<sos>"
EOS = "<eos>"

def read_file(filepath, num_sample):
    with open(filepath) as f:
        lines = f.read().split("\n")[:-1]
    line_count = 0
    english_text = []
    polish_text = []
    for line in lines:
        if line_count >= num_sample:
            return english_text, polish_text
        english, polish, _ = line.split("\t")
        polish = f"{SOS} " + polish + f" {EOS}"
        english_text.append(english)
        polish_text.append(polish)
        line_count += 1
    return english_text, polish_text

def split_data(english_text, polish_text):
    eng_train, eng_val, pol_train, pol_val = train_test_split(
        english_text, polish_text,
        test_size=0.3, random_state=42)
    eng_val, eng_test, pol_val, pol_test = train_test_split(
        eng_val, pol_val,
        test_size=0.33, random_state=42)
    return eng_train, eng_val, eng_test, pol_train, pol_val, pol_test

def create_vectorization(eng_train, pol_train, vocab_size):
    eng_len = max([len(line.split()) for line in eng_train])
    pol_len = max([len(line.split()) for line in pol_train])

    # We want to keep "<" and ">" that is used in SOS end EOS tokens
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")

    def custom_standardization(input_string):
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(
            lowercase, f"[{re.escape(strip_chars)}]", "")

    eng_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=eng_len,
    )
    pol_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=pol_len + 1, # +1, because we need to offset sequence during training
        standardize=custom_standardization
    )
    eng_vectorization.adapt(eng_train)
    pol_vectorization.adapt(pol_train)
    return eng_vectorization, pol_vectorization, eng_len, pol_len

def create_datasets(eng_train, eng_val, pol_train, pol_val, 
                    eng_vectorization, pol_vectorization,
                    batch_size):

    def format_dataset(eng, pol):
        eng = eng_vectorization(eng)
        pol = pol_vectorization(pol)
        return ({
            "encoder_inputs": eng,
            "decoder_inputs": pol[:, :-1], # input sentence do not have EOS token
        }, pol[:, 1:]) # target sentence is one step ahead

    def make_dataset(eng, pol):
        dataset = tf.data.Dataset.from_tensor_slices((eng, pol))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(format_dataset, num_parallel_calls=4)
        return dataset.shuffle(42).prefetch(16).cache()

    train_ds = make_dataset(eng_train, pol_train)
    val_ds = make_dataset(eng_val, pol_val)
    return train_ds, val_ds

# clean "ground truth" translations to match decoder output
def clean_sentence(input_sentence):
    input_sentence = input_sentence.replace(SOS, '')
    input_sentence = input_sentence.replace(EOS, '')
    input_sentence = input_sentence.translate(str.maketrans('', '', string.punctuation))
    input_sentence = input_sentence.lower()
    input_sentence = input_sentence.strip()
    return input_sentence