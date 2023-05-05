import os
import argparse
import logging
import math
import random
import time
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.utils.checkpoint

import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from huggingface_hub import create_repo, upload_folder
from diffusers.utils import check_min_version, is_wandb_available

import jax
import jax.numpy as jnp
import numpy as np
import optax


from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state, checkpoints
from flax.training.common_utils import shard
from flax.training import orbax_utils
import orbax.checkpoint

from PIL import Image, PngImagePlugin
from torch.utils.data import IterableDataset
from torchvision import transforms


import diffusion_model as dm
import model_utils as u


if is_wandb_available():
    import wandb

check_min_version("0.16.0.dev0")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--timesteps', type = int, default = 20000)
    parser.add_argument('--prefix', type = str, default = 'test')
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--latent_dim', type = int, default = 32)
    parser.add_argument('--seq_len', type = int, default = 64)
    parser.add_argument('--vocab_size', type = int, default = 821)

    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--report_to", type=str, default="wandb", help=('The integration to report the results and logs to. Currently only supported platforms are `"wandb"`'))
    parser.add_argument('--output_dir', type = str, default = 'test') 
    parser.add_argument('--hub_token', type = str, default = 'test')  # how do we get one?
    parser.add_argument('--hub_model_id', type = str, default = 'test') 
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1) 
    parser.add_argument("--profile_memory", action="store_true",  help="Whether to dump an initial (before training loop) and a final (at program end) memory profile.",)
    parser.add_argument( "--profile_steps", type=int, default=2,  help="How many training steps to profile in the beginning.",)
    parser.add_argument("--logging_steps", type=int, default=300, help=("log training metric every X steps to `--report_t`"),)
    parser.add_argument("--checkpointing_steps", type=int, default=300, help=("log training metric every X steps to `--report_t`"),)


    args = parser.parse_args()
    print('args', args)

    # sanity checks

    return args

# def validation, validation logging
# def save_model_card
# def make_train_dataset

def collate_fn(examples):
    input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples]).numpy()
    #hidden_states = jnp.stack([example["hidden_states"] for example in examples])
    batch = {
        "input_ids": input_ids,
            }
    return batch

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))



def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb init
    if jax.process_index() == 0 and args.report_to == "wandb":
        wandb.init(
            entity="diff-lm", # TODO add a args   
            project="diff-lm",
            job_type="train",
            config=args,
        )

    # init random keys
    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(args.seed)

    # init repo to push to HF
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # get tokeniser
    tokenizer = u.get_tokenizer()
    vocab_dict = u.make_vocab(tokenizer = tokenizer, vocab_path = 'vocab.json')
    args.vocab_size = len(vocab_dict)
    total_train_batch_size = args.batch_size * jax.local_device_count() * args.gradient_accumulation_steps

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
    
    # get datasets & data loaders (use total_train_batch_size, )
    # load models
    rng, rng_params = jax.random.split(rng)
    diff_lm = dm.DiffusionLM(timesteps = args.timesteps,
                        latent_dim = args.latent_dim,
                        batch_size = args.batch_size,
                        seq_len = args.seq_len,
                        vocab_size = len(vocab_dict))

    for b in train_dataloader:
      break                    
    
    diff_lm_params = diff_lm.init(rng, b['input_ids'], rng_params)

    # init pipeline (for validation)
    # prep optimizer, LR scheduler
    tx = optax.adamw(learning_rate=args.learning_rate, b1=0.9, b2=0.999, eps=1e-6)
    # create state
    state = train_state.TrainState.create(apply_fn=diff_lm.__call__, params=diff_lm_params, tx=tx)

    # init training
    validation_rng, train_rngs = jax.random.split(rng)
    train_rngs = jax.random.split(train_rngs, jax.local_device_count())


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
            loss, grad, new_train_rng = loss_and_grad(None, rng)
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

            loss, grad, new_train_rng = jax.lax.fori_loop(
                0, # from ind
                args.gradient_accumulation_steps, # to ind
                cumul_grad_step, # function
                init_loss_grad_rng, # data to apply function to 
            )
            loss, grad = jax.tree_map(lambda x: x / args.gradient_accumulation_steps, (loss, grad))

        ## done with grad accumulation ##

        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss} # maybe log each of the losses separately
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        rng,  new_train_rng = jax.random.split(rng)

        return new_state, new_train_rng, metrics # will be different from the GPU version!



    p_train_step = jax.pmap(train_step, "batch", ) # donate_argnums=(0,) we probably don't need it?

    state = jax_utils.replicate(state)

    # Train!
    dataset_length = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(dataset_length / args.gradient_accumulation_steps)

    max_train_steps = args.epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * num_update_steps_per_epoch}")

    if jax.process_index() == 0 and args.report_to == "wandb":
        wandb.define_metric("*", step_metric="train/step")
        wandb.define_metric("train/step", step_metric="walltime")
        wandb.config.update(
            {
                "num_train_examples": len(train_dataset),
                "total_train_batch_size": total_train_batch_size,
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
    
    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.output_dir, "memory_initial.prof"))
    t00 = t0 = time.monotonic()

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    #checkpoint_manager = orbax.checkpoint.CheckpointManager('managed_ckpts', orbax_checkpointer, options)
    checkpoint_manager = orbax.checkpoint.CheckpointManager('managed_ckpts2', orbax_checkpointer, options)
    save_args = orbax_utils.save_args_from_target(state)
    ckpt = {'model': state, 'config': {"name": "test"}, 'data': [b['input_ids']]}
    for epoch in epochs:

        train_metrics = []
        train_metric = None

        steps_per_epoch = (len(train_dataset) // total_train_batch_size)

        train_step_progress_bar = tqdm(
            total=steps_per_epoch,
            desc="Training...",
            position=1,
            leave=False,
            disable=jax.process_index() > 0,
        )

    
        # train
        for batch in train_dataloader:
            batch = batch['input_ids']
            if args.profile_steps and global_step == 1:
                train_metric["loss"].block_until_ready()
                jax.profiler.start_trace(args.output_dir)
            if args.profile_steps and global_step == 1 + args.profile_steps:
                train_metric["loss"].block_until_ready()
                jax.profiler.stop_trace()

            batch = shard(batch)
            with jax.profiler.StepTraceAnnotation("train", step_num=global_step):
                state, train_rngs, train_metric = p_train_step(state, batch, train_rngs)

            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= max_train_steps:
                break

            # add validation

            if global_step % args.logging_steps == 0 and jax.process_index() == 0:
                if args.report_to == "wandb":
                    train_metrics = jax_utils.unreplicate(train_metrics)
                    train_metrics = jax.tree_util.tree_map(lambda *m: jnp.array(m).mean(), *train_metrics)
                    wandb.log(
                        {
                            "walltime": time.monotonic() - t00,
                            "train/step": global_step,
                            "train/epoch": global_step / dataset_length,
                            "train/steps_per_sec": (global_step - step0) / (time.monotonic() - t0),
                            **{f"train/{k}": v for k, v in train_metrics.items()},
                        }
                    )
                t0, step0 = time.monotonic(), global_step
                train_metrics = []
            if global_step % args.checkpointing_steps == 0 and jax.process_index() == 0:
                # controlnet.save_pretrained(f"{args.output_dir}/{global_step}",
                #     params=get_params_to_save(state.params),)
                #ckpt = {'model': state, 'config': {"name": "test"}, 'data': batch}
                checkpoint_manager.save(global_step, state, save_kwargs = {"save_args": save_args})
                logger.info(f'Saved checkpoint at step {global_step}')
                # checkpoints.save_checkpoint(ckpt_dir=os.path.join(args.output_dir, args.prefix), target=state, step=global_step, keep = 2)

        # done with epoch
        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.epochs} | Loss: {train_metric['loss']})")

    #done with all epochs
    if jax.process_index() == 0:
        # final validation
        # and save
        # checkpoints.save_checkpoint(ckpt_dir=os.path.join(args.output_dir, args.prefix), target=state, step=global_step, keep = 2)
        #ckpt = {'model': state, 'config': {"name": "test"}, 'data': batch}
        checkpoint_manager.save(global_step, state, save_kwargs = {"save_args": save_args})
        logger.info(f'Saved final checkpoint at step {global_step}')

        if args.push_to_hub:
            # save_model_card(
            #     repo_id,
            #     image_logs=image_logs,
            #     base_model=args.pretrained_model_name_or_path,
            #     repo_folder=args.output_dir,
            # )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.output_dir, "memory_final.prof"))
    logger.info("Finished training.")

if __name__ == "__main__":
    main()
