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
  hidden_size : int = 768
  config = AutoConfig.from_pretrained('bert-base-cased')
  config.hidden_size = hidden_size
  position_ids = jnp.expand_dims(jnp.arange((config.max_position_embeddings)), 0)

  def setup(self):
    self.time_embed = FlaxTimestepEmbedding(self.hidden_size)
    self.timestep_embedding = FlaxTimesteps()
    self.input_up_proj = nn.Sequential([nn.Dense(self.hidden_size), nn.hard_tanh, nn.Dense(self.hidden_size)])
    self.position_embeddings = nn.Embed(self.config.max_position_embeddings, self.hidden_size)
    self.input_transformer = FlaxBertEncoder(self.config)
    self.layernorm = nn.LayerNorm()
    #self.dropout = nn.Dropout(self.config.hidden_dropout_prob, deterministic = False)
    self.output_down_proj = nn.Sequential([nn.Dense(self.hidden_size), nn.hard_tanh, nn.Dense(self.latent_dim)])
    self.lm_head = nn.Dense(self.vocab_size)

  
  def __call__(self, x, timesteps, y=None, src_ids=None, src_mask=None):
    
    emb = self.time_embed(self.timestep_embedding(timesteps))
    emb_x = self.input_up_proj(x)

    emb = jnp.broadcast_to(jnp.expand_dims(emb, 1), emb_x.shape)

    position_ids = self.position_ids[:, : self.seq_len ]

    input_emb = self.position_embeddings(position_ids) + emb_x + emb
    #input_emb = self.dropout(self.layernorm(input_emb))
    input_emb = self.layernorm(input_emb)
    input_trans_hidden_states = self.input_transformer(input_emb, None, None).last_hidden_state # attn_mask, head_mask!
    h = self.output_down_proj(input_trans_hidden_states)

    return h
  
