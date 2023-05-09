import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

import numpy as np


from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.models.modeling_flax_utils import FlaxModelMixin

from transformers import AutoConfig
from transformers import FlaxBertModel
from transformers.models.bert.modeling_flax_bert import FlaxBertEncoder
import torch

import transformer
import model_utils as u


class DiffusionLM(nn.Module):
  timesteps : int = 2000
  latent_dim : int = 32
  hidden_size : int = 768
  batch_size : int = 16
  seq_len : int = 64
  beta_schedule : str = 'linear'
  vocab_size : int = 333
  train : bool = True

  def setup(self):

    self.embedder = nn.Embed(self.vocab_size, self.latent_dim)
    self.transformer = transformer.Flax1DTransformer(latent_dim = self.latent_dim, seq_len = self.seq_len, vocab_size = self.vocab_size, hidden_size = self.hidden_size, train = self.train)
    #self.scheduler = FlaxDDPMScheduler(num_train_timesteps = self.timesteps, beta_start = 0.0001, beta_end =  0.02, beta_schedule = self.beta_schedule)
    #self.noise_scheduler_state = self.scheduler.create_state()
    self.get_alphas()

  
  def call_embedder(self, inp):
    return self.embedder(inp)
  
  def call_transformer(self, x, t):
    return self.transformer(x, t)
  

  # def init_weights(self):

  #   sample = jnp.zeros((self.batch_size, self.seq_len, self.latent_dim), dtype=jnp.float32)
  #   timesteps = jnp.ones((self.batch_size,), dtype=jnp.int32)

  #   params_rng, dropout_rng = jax.random.split(rng)
  #   rngs = {"params": params_rng, "dropout": dropout_rng}

  #   return self.init(rngs, sample, timesteps)['params']  # timesteps


  # def __call__(self, x, timesteps, rng : jax.random.PRNGKey = None):
  #   latents = self.embedder(x)

  #   rng, noise_rng = jax.random.split(rng)
  #   noise = jax.random.normal(noise_rng, latents.shape)
  #   rng, timestep_rng =  jax.random.split(noise_rng)
  #   timesteps = jax.random.randint(timestep_rng, (self.batch_size,), 0, self.scheduler.config.num_train_timesteps,)

  #   noisy_latents = self.scheduler.add_noise(self.noise_scheduler_state, latents, noise, timesteps)
  #   model_pred = self.transformer(noisy_latents, timesteps)

  #   loss = (noise - model_pred) ** 2

  #   rng, sample_rng =  jax.random.split(rng)
  #   t, weights = self.sample(sample_rng)

  #   return loss.mean()

  def get_logits(self, x):
    return self.transformer.lm_head(x)

  def __call__(self, x, rng : jax.random.PRNGKey = None):

    rng, sample_rng = jax.random.split(rng)

    t, weights = self.schedule_sampler(sample_rng, x.shape[0])

    x_start_mean = self.embedder(x)
    std = u.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, jnp.array([0]), x_start_mean.shape)

    rng, noise_rng = jax.random.split(rng)
    x_start = self.get_x_start(x_start_mean, std, noise_rng)

    rng, noise_rng2 = jax.random.split(noise_rng)
    noise = jax.random.normal(noise_rng, x_start.shape)
      
    x_t = self.q_sample(x_start, t, noise=noise) # (16, 64, 32) reparametrization trick.

    #get_logits = lambda x: self.transformer.lm_head(x)

    terms = {}

    model_output = self.transformer(x_t, t)

    terms["mse"] = u.mean_flat((x_start - model_output) ** 2)

    t0_mask = t == 0
    t0_loss = u.mean_flat((x_start_mean - model_output) ** 2)
    terms["mse"] = jnp.where(t0_mask, t0_loss, terms["mse"])

    out_mean, _, _ = self._q_mean_variance(x_start, jnp.array([self.timesteps - 1]))
    tT_loss = u.mean_flat(out_mean**2)

    decoder_nll = self.token_discrete_loss(x_start, x)

    terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)

    return terms["loss"]

  def schedule_sampler(self, rng, bsz):

      w = jnp.ones([self.timesteps])
      p = w / jnp.sum(w)
      indices = jax.random.choice(rng, len(p), shape=(bsz,), p=p)
      weights = 1 / (len(p) * p[indices])

      return indices, weights


  def get_std(self, timesteps, broadcast_shape):

    res = self.alphas[timesteps]
    while len(res.shape) < len(broadcast_shape):
      res = res[..., None]

    return jnp.broadcast_to(res, broadcast_shape)


  def get_x_start(self, x_start_mean, std, noise_rng):
    noise = jax.random.normal(noise_rng, x_start_mean.shape)
    return x_start_mean + std * noise


  def q_sample(self, x_start, t, noise):
    return (u.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
            u.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    

  def get_alphas(self):

    self._betas_for_alpha_bar()

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
    self.sqrt_alphas_cumprod =  jnp.sqrt(self.alphas_cumprod)
    self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
    self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
    self.posterior_mean_coef1 = self.betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * jnp.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    self.posterior_log_variance_clipped = jnp.log(jnp.append(self.posterior_variance[1], self.posterior_variance[1:]))

    return 


  def _betas_for_alpha_bar(self, max_beta = 0.999):

    schedule_fn = lambda t: 1 - jnp.sqrt(t + 0.0001)
    
    t1_arr = jnp.array(range(self.timesteps)) 
    t2_arr = t1_arr + 1

    t1_arr = t1_arr / self.timesteps
    t2_arr = t2_arr / self.timesteps

    betas = 1 - schedule_fn(t2_arr) / schedule_fn(t1_arr)

    self.betas = jnp.where(betas < max_beta, betas, max_beta)
  

  def _q_mean_variance(self, x_start, t):

    mean = u.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = u.extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
    log_variance = u.extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

    return mean, variance, log_variance


  def token_discrete_loss(self, x_t, input_ids):
      logits = self.get_logits(x_t)  # bsz, seqlen, vocab

      return u.crossEntropy(logits, input_ids)


  def q_posterior_mean_variance(self, x_start, x_t, t):

      assert x_start.shape == x_t.shape
      posterior_mean = (
          u.extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
          + u.extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
      )
      posterior_variance = u.extract_into_tensor(self.posterior_variance, t, x_t.shape)
      posterior_log_variance_clipped = u.extract_into_tensor(
          self.posterior_log_variance_clipped, t, x_t.shape
      )
      assert (
          posterior_mean.shape[0]
          == posterior_variance.shape[0]
          == posterior_log_variance_clipped.shape[0]
          == x_start.shape[0]
      )
      return posterior_mean, posterior_variance, posterior_log_variance_clipped
  

  def p_mean_variance(self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):

    # model is the transformer

    if model_kwargs is None:
      model_kwargs = {}

    B, C = x.shape[0], x.shape[-1]
    # B -> batch size, C -> channel size (embedding size)
    assert t.shape == (B,)

    model_output = self.transformer(x, t, **model_kwargs) # t -> self._scale_timesteps(t)
    model_variance = u.extract_into_tensor(self.posterior_variance, t, x.shape)
    model_log_variance = u.extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

    #pred_xstart = u.process_xstart(model_output)
    pred_xstart = model_output


    model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

    assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
    }
  

  def p_sample(
      self,
      x,
      t,
      rng,
      clip_denoised=True,
      denoised_fn=None,
      model_kwargs=None,
      top_p=None,
  ):
      """
      Sample x_{t-1} from the model at the given timestep.
      :param model: the model to sample from.
      :param x: the current tensor at x_{t-1}.
      :param t: the value of t, starting at 0 for the first diffusion step.
      :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
      :param denoised_fn: if not None, a function which applies to the
          x_start prediction before it is used to sample.
      :param model_kwargs: if not None, a dict of extra keyword arguments to
          pass to the model. This can be used for conditioning.
      :return: a dict containing the following keys:
                - 'sample': a random sample from the model.
                - 'pred_xstart': a prediction of x_0.
      """
      out = self.p_mean_variance(
          x,
          t,
          clip_denoised=clip_denoised,
          denoised_fn=denoised_fn,
          model_kwargs=model_kwargs,
      )

      if top_p is not None and top_p > 0:

        rng, noise_rng = jax.random.split(rng)
         
        noise = jax.random.normal(noise_rng, x.shape)

        # doesn't seem to work the same in JAX

        # def pos_noise(noise, noise_rng):

        #   replace_mask = jnp.abs(noise) > top_p
        #   if replace_mask.any():
        #     rng, noise_rng = jax.random.split(noise_rng)
        #     # noise[replace_mask] = jax.random.normal(noise_rng, x.shape) x = x.at[idx].set(y)
        #     noise = noise.at[replace_mask].set(jax.random.normal(noise_rng, x.shape))
        #     return pos_noise(noise, noise_rng)
        #   else:
        #     return noise

        # noise = pos_noise(noise, noise_rng)

        # no noise when t == 0
        t = jnp.where(t != 0, 1.0, 0.0)
        nonzero_mask = jnp.reshape(t, (-1, *([1]* (len(x.shape)-1) )))

        sample = out["mean"] + nonzero_mask * jax.lax.exp(0.5 * out["log_variance"]) * noise

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"],
            "out": out,}
      


  def p_sample_loop_progressive(
      self,
      shape,
      rng,
      noise=None,
      clip_denoised=True,
      denoised_fn=None,
      model_kwargs=None,
      device=None,
      progress=False,
      top_p=None,
      langevin_func=None,
  ):
    
    indices = list(range(self.timesteps))[::-1] # inference timesteps!!!

    if noise is not None:
        data = noise
    else:
        rng, noise_rng = jax.random.split(rng)
        data = jax.random.normal(noise_rng, shape)

    for i in indices:
        t = np.array([i] * shape[0])

        out = self.p_sample(
            data,
            t,
            rng,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            top_p=top_p,)
        yield out
        data = out["sample"]
        