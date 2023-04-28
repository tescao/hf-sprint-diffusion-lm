import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.models.modeling_flax_utils import FlaxModelMixin

from transformers import AutoConfig
from transformers import FlaxBertModel
from transformers.models.bert.modeling_flax_bert import FlaxBertEncoder
import torch

from .transformer import Flax1DTransformer
from .model_utils import extract_into_tensor, mean_flat, crossEntropy


class DiffusionLM(nn.Module):
  timesteps : int = 2000
  latent_dim : int = 32
  hidden_dim : int = 768
  batch_size : int = 16
  seq_len : int = 64
  beta_schedule : str = 'linear'
  vocab_size : int = 333

  def setup(self):

    #self.embedder = Embedder()
    self.embedder = nn.Embed(self.vocab_size, self.latent_dim)
    self.transformer = Flax1DTransformer(vocab_size = self.vocab_size)
    #self.scheduler = FlaxDDPMScheduler(num_train_timesteps = self.timesteps, beta_start = 0.0001, beta_end =  0.02, beta_schedule = self.beta_schedule)
    #self.noise_scheduler_state = self.scheduler.create_state()
    self.alphas_cumprod, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.log_one_minus_alphas_cumprod = self.get_alphas()# shape (2000,)

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

  def __call__(self, x, rng : jax.random.PRNGKey = None):

    rng, sample_rng = jax.random.split(rng)

    t, weights = self.schedule_sampler(sample_rng)

    x_start_mean = self.embedder(x)
    std = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, jnp.array([0]), x_start_mean.shape)

    rng, noise_rng = jax.random.split(rng)
    x_start = self.get_x_start(x_start_mean, std, noise_rng)

    rng, noise_rng2 = jax.random.split(noise_rng)
    noise = jax.random.normal(noise_rng, x_start.shape)
      
    x_t = self.q_sample(x_start, t, noise=noise) # (16, 64, 32) reparametrization trick.

    get_logits = lambda x: self.transformer.lm_head(x)

    terms = {}

    model_output = self.transformer(x_t, t)

    terms["mse"] = mean_flat((x_start - model_output) ** 2)

    t0_mask = t == 0
    t0_loss = mean_flat((x_start_mean - model_output) ** 2)
    terms["mse"] = jnp.where(t0_mask, t0_loss, terms["mse"])

    out_mean, _, _ = self._q_mean_variance(x_start, jnp.array([self.timesteps - 1]))
    tT_loss = mean_flat(out_mean**2)

    decoder_nll = self.token_discrete_loss(x_start, get_logits, x)

    terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)

    return terms["loss"]

  def schedule_sampler(self, rng):

      w = jnp.ones([self.timesteps])
      p = w / jnp.sum(w)
      indices = jax.random.choice(rng, len(p), shape=(self.batch_size,), p=p)
      weights = 1 / (len(p) * p[indices])

      return indices, weights


  def get_std(self, timesteps, broadcast_shape):

    res = self.alphas[timesteps]
    while len(res.shape) < len(broadcast_shape):
      print(res.shape)
      res = res[..., None]

    return jnp.broadcast_to(res, broadcast_shape)


  def get_x_start(self, x_start_mean, std, noise_rng):
    noise = jax.random.normal(noise_rng, x_start_mean.shape)
    return x_start_mean + std * noise


  def q_sample(self, x_start, t, noise):
    return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    

  def get_alphas(self):

    self._betas_for_alpha_bar()

    alphas = 1.0 - self.betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    sqrt_alphas_cumprod =  jnp.sqrt(alphas_cumprod)
    log_one_minus_alphas_cumprod = jnp.log(1.0 - alphas_cumprod)

    return alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod


  def _betas_for_alpha_bar(self, max_beta = 0.999):
    # returns betas, only for sqrt schedule
    schedule_fn = lambda t: 1 - jnp.sqrt(t + 0.0001)

    betas = []
    for i in range(self.timesteps):
        t1 = i / self.timesteps
        t2 = (i + 1) / self.timesteps
        betas.append(min(1 - schedule_fn(t2) / schedule_fn(t1), max_beta))

    self.betas = jnp.array(betas)



  def _q_mean_variance(self, x_start, t):

    mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
    log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

    return mean, variance, log_variance


  def token_discrete_loss(self, x_t, get_logits, input_ids):
      logits = get_logits(x_t)  # bsz, seqlen, vocab

      return crossEntropy(logits, input_ids)
