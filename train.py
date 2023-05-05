# import locale
# locale.getpreferredencoding = lambda *args: "UTF-8"

import os
import argparse
import logging
import math
import time
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.utils.checkpoint

import transformers

from datasets import load_dataset, load_from_disk
from transformers import set_seed
from huggingface_hub import create_repo, upload_folder
from diffusers.utils import check_min_version, is_wandb_available

import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax

from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state, checkpoints
from flax.training.common_utils import shard
from flax.training import orbax_utils
import orbax.checkpoint

import diffusion_model as dm
import model_utils as u

if is_wandb_available():
    import wandb

check_min_version("0.16.0.dev0")

print(f"Device count : {jax.device_count()}")

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
    parser.add_argument('--batch_size', type = int, default = 16) # set bigger, think batch_size * accum_steps 
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--timesteps', type = int, default = 2000)
    parser.add_argument('--prefix', type = str, default = 'test')
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--latent_dim', type = int, default = 32)
    parser.add_argument('--seq_len', type = int, default = 64)
    
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--report_to", type=str, default="wandb", help=('The integration to report the results and logs to. Currently only supported platforms are `"wandb"`'))
    parser.add_argument('--output_dir', type = str, default = 'test') 
    parser.add_argument('--hub_token', type = str, default = 'test')  # how do we get one?
    parser.add_argument('--hub_model_id', type = str, default = 'test') 
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1) 
    parser.add_argument("--profile_memory", action="store_true",  help="Whether to dump an initial (before training loop) and a final (at program end) memory profile.",)
    parser.add_argument("--profile_steps", type=int, default=2,  help="How many training steps to profile in the beginning.",)
    parser.add_argument("--logging_steps", type=int, default=300, help=("log training metric every X steps to `--report_t`"),)
    parser.add_argument("--checkpointing_steps", type=int, default=300, help=("log training metric every X steps to `--report_t`"),)

    args = parser.parse_args()
    print('args', args)
    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

    if jax.process_index() == 0 and args.report_to == "wandb":
        wandb.init(
            entity="diff-lm",  
            project="diff-lm",
            job_type="train",
            config=args,
        )


    # load data
    #vocab_path = 'vocab.json'
    tokenizer = u.get_tokenizer()
    vocab_dict = u.make_vocab(tokenizer = tokenizer, vocab_path = 'vocab.json')

    train_dataset = u.make_dataset('data/e2e_data/src1_train.txt', vocab_dict, padding_mode = 'block', seq_length = args.seq_len)
    test_dataset = u.make_dataset('data/e2e_data/src1_test.txt', vocab_dict, padding_mode = 'block', seq_length = args.seq_len)
    val_dataset = u.make_dataset('data/e2e_data/src1_valid.txt', vocab_dict, padding_mode = 'block', seq_length = args.seq_len)


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
    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(args.seed)
    rng, rng_params = jax.random.split(rng)
    rng, rng_dropout = jax.random.split(rng)

    diff_lm = dm.DiffusionLM(timesteps = args.timesteps,
                        latent_dim = args.latent_dim,
                        batch_size = args.batch_size,
                        seq_len = args.seq_len,
                        vocab_size = len(vocab_dict))

    for b in train_dataloader:
      break                    
    
    diff_lm_params = diff_lm.init({'params' : rng, 'dropout' : rng_dropout}, b['input_ids'], rng_params) # jnp.ones((args.batch_size, args.seq_len, args.latent_dim))

    # prep for training
    tx = optax.adamw(learning_rate=args.learning_rate, b1=0.9, b2=0.999, eps=1e-6)
    state = train_state.TrainState.create(apply_fn=diff_lm.__call__, params=diff_lm_params, tx=tx)
    train_rng, validation_rng = jax.random.split(rng)

    @jax.jit
    def train_step(state, batch, rng):
        if args.gradient_accumulation_steps > 1:
            grad_steps = args.gradient_accumulation_steps
            batch = jax.tree_map(lambda x: x.reshape((grad_steps, x.shape[0] // grad_steps) + x.shape[1:]), batch) # split into mini-batches

        def compute_loss(params, batch, rng):
            batch_losses = diff_lm.apply(params, batch, rng)
            return batch_losses.mean()

        grad_fn = jax.value_and_grad(compute_loss)

        ## for grad accumulation ##
        def get_minibatch(batch, grad_idx):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                batch)
        
        def loss_and_grad(grad_idx, train_rng):
            # create minibatch for the grad step
            minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            sample_rng, train_rng = jax.random.split(train_rng, 2)
            loss, grad = grad_fn(state.params, minibatch, sample_rng) # why does it need an rng?
            return loss, grad, train_rng
        
        if args.gradient_accumulation_steps == 1:
            loss, grads, new_train_rng = loss_and_grad(None, rng)
        else:
            init_loss_grad_rng = (
                0.0,  # initial value for cumul_loss
                jax.tree_map(jnp.zeros_like, state.params),  # initial value for cumul_grad
                rng,  # initial value for train_rng
            )

            def cumul_grad_step(grad_idx, loss_grad_rng):
                cumul_loss, cumul_grad, train_rng = loss_grad_rng
                loss, grad, new_train_rng = loss_and_grad(grad_idx, train_rng)
                cumul_loss, cumul_grad = jax.tree_map(jnp.add, (cumul_loss, cumul_grad), (loss, grad))
                return cumul_loss, cumul_grad, new_train_rng

            loss, grads, new_train_rng = jax.lax.fori_loop(
                0, # from ind
                args.gradient_accumulation_steps, # to ind
                cumul_grad_step, # function
                init_loss_grad_rng, # data to apply function to 
            )
            loss, grads = jax.tree_map(lambda x: x / args.gradient_accumulation_steps, (loss, grads))

        ## done with grad accumulation ##

        #loss, grads, new_train_rng = loss_and_grad(state.params, batch, rng)
        new_state = state.apply_gradients(grads=grads)

        return new_state, new_train_rng, loss

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_train_batch_size = args.batch_size
    max_train_steps = args.epochs * num_update_steps_per_epoch
    dataset_length = len(train_dataloader)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {total_train_batch_size}")
    logger.info(f"  Train batch per step = {args.batch_size // args.gradient_accumulation_steps}")
    logger.info(f"  Otimization steps per epochs= {args.epochs // args.batch_size}")

    if jax.process_index() == 0  and args.report_to == "wandb":
        wandb.define_metric("*", step_metric="train/step")
        wandb.define_metric("train/step", step_metric="walltime")
        wandb.config.update(
            {
                "num_train_examples": len(train_dataset),
                "total_train_batch_size": args.batch_size,
                "total_optimization_step": args.epochs * num_update_steps_per_epoch,
                "num_devices": jax.device_count(),
                "diffusion_lm_params": sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params)),
            }
        )

    global_step = step0 = 0
    epochs = tqdm(
        range(args.epochs),
        desc="Epoch ... ",
        position=0,
        disable=jax.process_index() > 0,)
    t00 = t0 = time.monotonic()

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager('managed_ckpts', orbax_checkpointer, options)
    save_args = orbax_utils.save_args_from_target(state)

    models_dir = 'models'
    for epoch in epochs:

        train_metrics = []
        train_metric = None

        steps_per_epoch = (len(train_dataset) // total_train_batch_size)

        train_step_progress_bar = tqdm(
            total=steps_per_epoch,
            desc="Training...",
            position=1,
            leave=False,
            disable=jax.process_index() > 0,)

        for batch in train_dataloader:
            state, train_rng, loss = train_step(state, batch['input_ids'], train_rng)

            train_metrics.append(loss)
            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= max_train_steps:
                break

            if global_step % args.logging_steps == 0 and jax.process_index() == 0:
                if args.report_to == "wandb":
                    train_metrics = jax.tree_util.tree_map(lambda *m: jnp.array(m).mean(), *train_metrics)
                    wandb.log(
                        {
                            "walltime": time.monotonic() - t00,
                            "train/step": global_step,
                            "train/epoch": global_step / dataset_length,
                            "train/steps_per_sec": (global_step - step0) / (time.monotonic() - t0),
                            **{"train/loss": train_metrics}, # **{f"train/{k}": v for k, v in train_metrics.items()}
                        }
                    )
                t0, step0 = time.monotonic(), global_step
                train_metrics = []

            if global_step % args.checkpointing_steps == 0 and jax.process_index() == 0:
                checkpoint_manager.save(global_step, state, save_kwargs = {"save_args": save_args})
                logger.info(f'Saved checkpoint at step {global_step}')

    train_step_progress_bar.close()
    epochs.write(f"Epoch... ({epoch + 1}/{args.epochs} | Loss: {loss})")

    if jax.process_index() == 0:
        checkpoint_manager.save(global_step, state, save_kwargs = {"save_args": save_args})
        logger.info(f'Saved final checkpoint at step {global_step}')

    
if __name__ == "__main__":
    main()
