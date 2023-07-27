from typing import Callable

import numpy as np
import jax
import flax.linen as nn
import jax.numpy as jnp

from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.models.modeling_flax_utils import FlaxModelMixin

from transformers import AutoConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertEncoder
  

class Flax1DTransformer(nn.Module, FlaxModelMixin):

  latent_dim : int = 32 
  seq_len : int = 64 
  vocab_size : int = 333

  config = AutoConfig.from_pretrained('bert-base-cased')
  position_ids = jnp.expand_dims(jnp.arange((config.max_position_embeddings)), 0)
  use_pretrained : bool = True
  train : bool =  True


  def setup(self):
    self.time_embed = FlaxTimestepEmbedding(self.config.hidden_size)
    self.timestep_embedding = FlaxTimesteps()

    self.position_embeddings = nn.Embed(self.config.max_position_embeddings, self.config.hidden_size)

    if not self.use_pretrained:
        self.input_up_proj = nn.Sequential([nn.Dense(self.latent_dim), nn.hard_tanh, nn.Dense(self.config.hidden_size)])
        self.output_down_proj = nn.Sequential([nn.Dense(self.config.hidden_size), nn.hard_tanh, nn.Dense(self.latent_dim)])

    self.input_transformer = FlaxBertEncoder(self.config)

    self.layernorm = nn.LayerNorm()
    self.dropout = nn.Dropout(self.config.hidden_dropout_prob, deterministic = not self.train)

    #self.output_down_proj = nn.Sequential([nn.Dense(self.hidden_size), nn.hard_tanh, nn.Dense(self.latent_dim)])
    #self.lm_head = nn.Dense(self.vocab_size)

  
  def __call__(self, x, timesteps, y=None, src_ids=None, src_mask=None):
    

    if not self.use_pretrained:
       x = self.input_up_proj(x)

    emb = self.time_embed(self.timestep_embedding(timesteps))
    emb = jnp.broadcast_to(jnp.expand_dims(emb, 1), x.shape)

    position_ids = self.position_ids[:, : self.config.max_position_embeddings ]

    # x [bsz, 512, 768]
    input_emb = self.position_embeddings(position_ids) + x + emb
    input_emb = self.dropout(self.layernorm(input_emb))
    h = self.input_transformer(input_emb, None, None).last_hidden_state # attn_mask, head_mask!

    if not self.use_pretrained:
       h = self.output_down_proj(h)

    return h
  


class FlaxBertPredictionHeadTransform(nn.Module):
  dtype : jnp.dtype = jnp.float32
  hidden_size : int = 768
  layer_norm_eps : float = 1e-6

  def setup(self):
      self.dense = nn.Dense(self.hidden_size, dtype=self.dtype)
      self.activation = nn.gelu
      self.LayerNorm = nn.LayerNorm(epsilon=self.layer_norm_eps, dtype=self.dtype)

  def __call__(self, hidden_states):
      hidden_states = self.dense(hidden_states)
      hidden_states = self.activation(hidden_states)
      return self.LayerNorm(hidden_states)


class FlaxBertLMPredictionHead(nn.Module):
    dtype : jnp.dtype = jnp.float32
    hidden_size : int = 768
    bias_init : Callable[..., np.ndarray] = jax.nn.initializers.zeros
    vocab_size : int = 333

    def setup(self):
        self.transform = FlaxBertPredictionHeadTransform(hidden_size = self.hidden_size, dtype=self.dtype)
        self.decoder = nn.Dense(self.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        #hidden_states = self.transform(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias

        return hidden_states
