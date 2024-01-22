from nnsight import LanguageModel
from buffer import ActivationBuffer
from training import trainSAE
from utils import hf_dataset_to_generator

model = LanguageModel(
    "EleutherAI/pythia-70m-deduped",  # this can be any Huggingface model
    device_map="cuda:0",
)
submodule = model.gpt_neox.layers[1].mlp  # layer 1 MLP
activation_dim = 512  # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data much be an iterator that outputs strings
data = hf_dataset_to_generator("monology/pile-uncopyrighted")
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    out_feats=activation_dim,  # output dimension of the model component
)  # buffer will return batches of tensors of dimension = submodule's output dimension

# train the sparse autoencoder (SAE)
ae = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    device="cuda:0",
    save_steps=10000,
    save_dir="",
    log_steps=100,
    steps=100000,
)
