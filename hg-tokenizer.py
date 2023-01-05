import os
import time

from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

spm = SentencePieceUnigramTokenizer.from_spm("iwslt14_en-de_spm_model")

tokenizer = PreTrainedTokenizerFast(tokenizer_object=spm)

# IWSLT model does not include pad token, so we manually add one.
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Load the entire example dataset into memory.
with open("example.txt") as fp:
    data = [line.rstrip() for line in fp.readlines()]

print("Tokenizing...")

s_time = time.perf_counter()

encoded_data = tokenizer(data, padding=True, return_tensors="pt")

e_time = time.perf_counter()

print(f"Done in {e_time - s_time} seconds!")

print(encoded_data.input_ids.shape)
