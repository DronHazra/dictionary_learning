from nnsight import LanguageModel
from buffer import ActivationBuffer
from training import trainSAE
from utils import hf_dataset_to_generator
import wandb

DEVICE = "mps"

MODEL = "EleutherAI/pythia-70m-deduped"
DATASET = "monology/pile-uncopyrighted"
LR = 3e-4
SPARSITY_PENALTY = 1e-3
model = LanguageModel(
    MODEL,  # this can be any Huggingface model
    device_map=DEVICE,
)
submodule = model.gpt_neox.layers[1].mlp  # layer 1 MLP
activation_dim = 512  # output dimension of the MLP
dictionary_size = 16 * activation_dim

wandb.init(project="sae", entity="dronh")
wandb.config.update(
    {
        "model": MODEL,
        "dataset": DATASET,
        "activation_dim": activation_dim,
        "dictionary_size": dictionary_size,
        "lr": LR,
        "sparsity_penalty": SPARSITY_PENALTY,
    }
)
# data much be an iterator that outputs strings
data = hf_dataset_to_generator(DATASET)
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
    lr=LR,
    sparsity_penalty=SPARSITY_PENALTY,
    device=DEVICE,
    save_steps=10000,
    save_dir="",
    log_steps=100,
    steps=100001,
)
