# import locale
# locale.getpreferredencoding = lambda *args: "UTF-8"

import os
import argparse
import logging

import torch
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from .diffusion_model import DiffusionLM
from .utils import make_vocab, make_dataset


logger = logging.getLogger(__name__)

def collate_fn(examples):
    input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples]).numpy()
    #hidden_states = jnp.stack([example["hidden_states"] for example in examples])
    batch = {
        "input_ids": input_ids,
            }
    return batch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--timesteps', type = int, default = 20000)
    parser.add_argument('--prefix', type = str, default = 'test')
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--latent_dim', type = int, default = 32)
    parser.add_argument('--seq_len', type = int, default = 64)

    args = parser.parse_args()
    print('args', args)
    return args


def main():
    args = parse_args()

    # load data
    #vocab_path = 'vocab.json'
    vocab_dict = make_vocab(vocab_path = 'vocab.json', rewrite = True)

    train_dataset = make_dataset('data/e2e_data/src1_train.txt', vocab_dict, padding_mode = 'block', seq_length = 64)
    test_dataset = make_dataset('data/e2e_data/src1_test.txt', vocab_dict, padding_mode = 'block', seq_length = 64)
    val_dataset = make_dataset('data/e2e_data/src1_valid.txt', vocab_dict, padding_mode = 'block', seq_length = 64)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True, # false if streaming dataset
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        drop_last=True)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False, 
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False, 
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        drop_last=True)


    # initialize
    rng = jax.random.PRNGKey(args.seed)
    rng, train_rng = jax.random.split(rng)

    diff_lm = DiffusionLM(timesteps = args.timesteps,
                        latent_dim = args.latent_dim,
                        batch_size = args.batch_size,
                        seq_len = args.seq_len,
                        vocab_size = len(vocab_dict))
    
    diff_lm_params = diff_lm.init(rng, jnp.ones((args.batch_size, args.seq_len, args.latent_dim)), train_rng)

    # prep for training
    tx = optax.adamw(learning_rate=args.learning_rate, b1=0.9, b2=0.999, eps=1e-6)
    state = train_state.TrainState.create(apply_fn=diff_lm.__call__, params=diff_lm_params, tx=tx)


    def train_step(state, batch, rng):

        def compute_loss(params, batch, rng):
            batch_losses = diff_lm.apply(params, batch['input_ids'], rng)
            return batch_losses.mean()

        train_rng,  new_train_rng = jax.random.split(rng)

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params, batch, rng)

        new_state = state.apply_gradients(grads=grads)

        return new_state,  new_train_rng, loss
    
    losses = []
    for ep in range(args.epochs):
        print('Epoch', ep)
        for batch in train_dataloader:
            state,  rng, loss = train_step(state, batch['input_ids'], rng)
            print('batch loss', loss)
            losses.append(loss)

        diff_lm.save_pretrained(f"models/{args.prefix}_ep{ep}", params=state.params,)


    
if __name__ == "__main__":
    main()

"""
ToDo

port to TPU
add sprint-related requirements (docstring, push to hub)
add evaluation
add wandb

add intermittent saving
add reloading model to continue training
add training stopping criteria

add other datasets
add different tokenizers (SP)

fix BERT inputs
add dropout

load vocab
add other padding modes (only block now)
"""