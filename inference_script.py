# model path = managed_ckpts/26000/default
# if loading on a CPU
import os
import argparse
import logging

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax

print(jax.devices())

from flax.core.frozen_dict import FrozenDict
import numpy as np
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze

from orbax.checkpoint import PyTreeCheckpointer, CheckpointManagerOptions, CheckpointManager

import diffusion_model as dm
import model_utils as u


class Args():
    seed : int = 0
    timesteps : int = 20000
    latent_dim : int = 32
    batch_size : int = 64
    seq_len : int = 64
    vocab_size : int = 821


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--step', type = int, default = 26000)
    parser.add_argument('--model_dir', type = str, default = 'managed_ckpts') 
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--diff_steps', type = int, default = 200)
    parser.add_argument('--top_p', type = float, default = 0.8) # what's a good default?
    parser.add_argument('--num_samples', type = int, default = 10)
    parser.add_argument('--timesteps', type = int, default = 200) # use to init diffusion model for inference


    args = parser.parse_args()
    print('args', args)

    # sanity checks

    return args

def main():

    args = parse_args()
    train_args = Args()

    tokenizer = u.get_tokenizer()
    vocab_dict = u.make_vocab(tokenizer = tokenizer, vocab_path = 'vocab.json')
    vocab_dict_r = u.get_decoder(vocab_dict)

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = CheckpointManager(args.model_dir, orbax_checkpointer, options)

    step = args.step if args.step else checkpoint_manager.latest_step()
    state_dict = checkpoint_manager.restore(step)

    print('Loaded state dict', state_dict.keys())
    print(state_dict['params']['params'].keys())


    rng = jax.random.PRNGKey(train_args.seed)
    rng, rng_params = jax.random.split(rng)
    diff_lm = dm.DiffusionLM(timesteps = args.timesteps,
                        latent_dim = train_args.latent_dim,
                        batch_size = train_args.batch_size,
                        seq_len = train_args.seq_len,
                        vocab_size = len(vocab_dict))
    
    inp = jnp.ones(shape = (train_args.batch_size, train_args.seq_len), dtype = jnp.int32)
    diff_lm_init_params = diff_lm.init(rng, inp, rng_params)

    diff_lm.get_alphas() # initialized all values

    diff_lm_init_params = unfreeze(diff_lm_init_params)

    diff_lm_init_params['params']['transformer']

    print('emb shape' , diff_lm_init_params['params']['embedder']['embedding'].shape)
    print('lm head shape' , diff_lm_init_params['params']['transformer']['lm_head']['kernel'].shape)

    print('diff_lm_init_params')
    for k in diff_lm_init_params['params']['transformer']:
        for kk in diff_lm_init_params['params']['transformer'][k]:
            if not isinstance(diff_lm_init_params['params']['transformer'][k][kk], (FrozenDict, dict)):
                diff_lm_init_params['params']['transformer'][k][kk] = state_dict['params']['params']['transformer'][k][kk][0]
            else:
                for kkk in diff_lm_init_params['params']['transformer'][k][kk]:
                    if not isinstance(diff_lm_init_params['params']['transformer'][k][kk][kkk], (FrozenDict, dict)):
                        diff_lm_init_params['params']['transformer'][k][kk][kkk] = state_dict['params']['params']['transformer'][k][kk][kkk][0]
                    else:
                        for kkkk in diff_lm_init_params['params']['transformer'][k][kk][kkk]:
                            if not isinstance(diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk], (FrozenDict, dict)):
                                diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk] = state_dict['params']['params']['transformer'][k][kk][kkk][kkkk][0]
                            else:
                                for kkkkk in diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk]:
                                    if not isinstance(diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk], (FrozenDict, dict)):
                                        diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk] = state_dict['params']['params']['transformer'][k][kk][kkk][kkkk][kkkkk][0]
                                    else:
                                        for kkkkkk in diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk]:
                                            if not isinstance(diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk], (FrozenDict, dict)):
                                                diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk] = state_dict['params']['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk][0]
                                            else:
                                                for kkkkkkk in diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk]:
                                                    if not isinstance(diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk][kkkkkkk], (FrozenDict, dict)):
                                                        diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk][kkkkkkk] = state_dict['params']['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk][kkkkkkk][0]
                                                    else:
                                                        print('last, unchanged ', diff_lm_init_params['params']['transformer'][k][kk][kkk][kkkk][kkkkk][kkkkkk][kkkkkkk])                               


    diff_lm_init_params = freeze(diff_lm_init_params)


    print('diff_lm_init_params, initialized')

    # testing
    # embeddings = diff_lm.apply(diff_lm_init_params, inp, method = diff_lm.call_embedder)
    # print('embeddings shape', embeddings.shape)

    # transformer_fn = lambda x,t : diff_lm.apply(diff_lm_init_params, x, t, method = diff_lm.call_transformer)
    # print('Calling transformer_fn')
    # res = transformer_fn(embeddings, jnp.ones( shape = (embeddings.shape[0]), dtype = jnp.int32))
    # print(res.shape)

    lm_head_fn = lambda x: diff_lm.apply(diff_lm_init_params, x, method = diff_lm.get_logits)

    
    def p_sample_loop(
        shape,
        rng,
        loop_fn,
        lm_head_fn,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        tokenizer=None,
        log_verbose=False,
        logging_freq: int = 100,
        num_samples_to_show: int = 10,
        langevin_fn=None,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        prevs = [[] for _ in range(num_samples_to_show)]

        for i, sample in enumerate(
            loop_fn(
                shape,
                rng,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                top_p=top_p,
                langevin_func=langevin_fn,
            )
        ):
            final = sample

            # if i % logging_freq == 0 and log_verbose:
            #     x_t = sample["sample"]
            #     logits = lm_head_fn(x_t)  # bsz, seqlen, vocab
            #     cands = jax.lax.top_k(logits, 1)

            #     for j in range(num_samples_to_show):
            #         ids = cands.indices[j].squeeze(1).tolist()
            #         improved_sent = " ".join([tokenizer[x[0].item()] for x in ids])
            #         prevs[j].append(f"[step {i}] " + improved_sent)
            #         u.pprint_sentences(
            #             sentences=prevs[j],
            #             banner=f"DDPM Denoising Step = {i} | Sample #{j + 1}",
            #             sep=" -> ",
            #         )
            # if i == diff_lm.timesteps - diff_lm.timesteps // 10:
            #     return final['sample']

        return final["sample"]


    loop_fn = lambda shape, rng, **kwargs : diff_lm.apply(diff_lm_init_params, shape, rng, **kwargs, method =  diff_lm.p_sample_loop_progressive)

    all_samples = []
    while len(all_samples) * args.batch_size < args.num_samples:
        sample_shape = (args.batch_size, train_args.seq_len, train_args.latent_dim)
        rng, sample_rng = jax.random.split(rng)
        sample = p_sample_loop(sample_shape,
                               sample_rng,
                               loop_fn,
                               lm_head_fn,
                               noise = None,
                               denoised_fn=None,
                               top_p = args.top_p,
                               tokenizer=tokenizer,
                               )
        
        all_samples.append(sample)

    arr = jnp.concatenate(all_samples, axis=0)
    arr = arr[:args.num_samples]

    logits = lm_head_fn(arr) # transformer
    print(logits.shape)
    cands, inds = jax.lax.top_k(logits, 1) 

    decoded_sentences = []
    for seq in inds:
        decoded_sentence =  " ".join([vocab_dict_r[x.item()] for x in seq])
        print(decoded_sentence)
        decoded_sentences.append(decoded_sentence)

    with open('out.txt', 'w') as f:
        f.write('\n'.join(decoded_sentences))


if __name__ == "__main__":
    main()