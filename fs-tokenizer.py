import os
import time

import torch
from torch import Tensor

from fairseq2.data import DataPipeline, StringLike, list_files, zip_data_pipelines, read_sequence
from fairseq2.data.text import SentencePieceEncoder, SentencePieceModel, read_text

# For demonstration purposes, we allocate our batches on the first CUDA
# device.
#device = torch.device("cuda:0")
#
device = torch.device("cpu")


# This is our SentencePiece model API implemented in C++.
spm = SentencePieceModel(
    pathname="iwslt14_en-de_spm_model",
    # The `control_tokens` parameter is used to natively add custom tokens (e.g.
    # pad, language markers) directly at the Protobuf level. Here we add <pad>
    # since the IWSLT SentencePiece model does not have a pad token.
    control_tokens=["<pad>"],
)

# Unlike the official SentencePiece API we refactored our encoding/decoding
# API from the actual model API.
encoder = SentencePieceEncoder(
    # Use the SentencePiece model we just instantiated.
    spm,
#    # Enable sampling (a.k.a. regulazation).
#    enable_sampling=True,
#    # These are the default values for `nbest_size` and `alpha`, we specify
#    # them here for demonstration purposes.
#    nbest_size=1,
#    alpha=0.1,
#    # If the number of input text lines is less than 16, pad the batch
#    # vertically.
#    batch_size=16,
    # We support int16, int32, and int64 (int8 coming later). Since we want to
    # have a fair comparison, we use int64 like HG tokenizers.
    dtype=torch.int64,
    # We are using the lowest-level SentencePiece API which makes it
    # possible for us to tokenize the text directly into the tensor storage.
    device=device,
#    # We pin the host memory to speed up the GPU data transfer after
#    # tokenization.
#    pin_memory=True,
    disable_parallelism=False,
)

# Load the entire example dataset into memory.
with open("example.txt") as fp:
    data = fp.readlines()

print("Tokenizing...")

s_time = time.perf_counter()

encoded_data = encoder(data)

e_time = time.perf_counter()

print(f"Done in {e_time - s_time} seconds!")

print(encoded_data.shape)
