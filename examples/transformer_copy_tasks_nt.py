"""An example doing inference with an infinitely wide attention network on IMDb.

Adapted from
https://github.com/google/neural-tangents/blob/main/examples/infinite_fcn.py

By default, this example does inference on a very small subset, and uses small
word embeddings for performance. A 300/300 train/test split takes 30 seconds
on a machine with 2 Titan X Pascal GPUs, please adjust settings accordingly.

For details, please see "`Infinite attention: NNGP and NTK for deep attention
networks <https://arxiv.org/abs/2006.10540>`_".
"""

import time
import sys; import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/examples")

from absl import app
from jax import random
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax
import util
import transformer_datasets as TD


_BATCH_SIZE = 15  # Batch size for kernel computation. 0 for no batching.
_MAX_SENTENCE_LENGTH = 500  # Pad/truncate sentences to this length.



def get_task_data(task, level='easy', dataset_size: int=1000, mask_contstant=100.):
    if level == 'easy':
        num_letters = 8; max_input_len = 16
    elif level == 'hard':
        num_letters = 26; max_input_len = 32
    if task == 'copy':
        dataset = TD.CopyTaskDataset(dataset_size=dataset_size,
                                     num_letters=num_letters,
                                     max_input_len=max_input_len)
    elif task == 'first':
        dataset = TD.SelectTokenTaskDataset(dataset_size=dataset_size,
                                            num_letters=num_letters,
                                            max_input_len=max_input_len,
                                            which_token='first')
    elif task == 'last':
        dataset = TD.SelectTokenTaskDataset(dataset_size=dataset_size,
                                            num_letters=num_letters,
                                            max_input_len=max_input_len,
                                            which_token='last')
    elif task == 'suffix-lookup':
        dataset = TD.SuffixKeyLookupTaskDataset(dataset_size=dataset_size,
                                                num_letters=num_letters,
                                                max_input_len=max_input_len)
    else:
        raise NotImplementedError(f"task {task} not implemented")

    for i in range(dataset_size):
      x = dataset.__getitem__(i)
    # TODO: stack data points x into a tensor of size (B, S, d)
    # TODO: stack y into (B, C)
    # TODO fix a positional encoding and token encoding
    #
    breakpoint()


def main(*args, **kwargs) -> None:
  # Mask all padding with this value.
  mask_constant = 100.
  get_task_data('first')
  x_train, y_train, x_test, y_test = _get_dummy_data(mask_constant)
  breakpoint()

  # Build the infinite network.
  # Not using the finite model, hence width is set to 1 everywhere.
  _, _, kernel_fn = stax.serial(
      stax.Dense(out_dim=1),
      stax.Relu(),
      stax.GlobalSelfAttention(
          n_chan_out=1,
          n_chan_key=1,
          n_chan_val=1,
          pos_emb_type='SUM',
          W_pos_emb_std=1.,
          pos_emb_decay_fn=lambda d: 1 / (1 + d**2),
          n_heads=1),
      stax.Relu(),
      # stax.GlobalAvgPool(),
      stax.Dense(out_dim=1)
  )

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=_BATCH_SIZE)

  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  predict = nt.predict.gradient_descent_mse_ensemble(
      kernel_fn=kernel_fn,
      x_train=x_train,
      y_train=y_train,
      diag_reg=1e-6,
      mask_constant=mask_constant)

  fx_test_nngp, fx_test_ntk = predict(x_test=x_test, get=('nngp'))

  fx_test_nngp.block_until_ready()
  fx_test_ntk.block_until_ready()

  duration = time.time() - start
  print(f'Kernel construction and inference done in {duration} seconds.')

  # Print out accuracy and loss for infinite network predictions.
  loss = lambda fx, y_hat: 0.5 * jnp.mean((fx - y_hat) ** 2)
  util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
  util.print_summary('NTK test', y_test, fx_test_ntk, None, loss)


def _get_dummy_data(
    mask_constant: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Return dummy data for when downloading embeddings is not feasible."""
  n_train, n_test = 6, 6

  def get_x(shape, key):
    key_x, key_mask = random.split(key)
    x = random.normal(key_x, shape)
    mask = random.bernoulli(key_mask, 0.6, shape)
    x = jnp.where(mask, mask_constant, x)
    return x

  def get_y(x):
    x = jnp.where(x == mask_constant, 0., x)

    def weighted_sum(x, start, end):
      return jnp.sum(x[..., start:end] *
                     jnp.arange(x.shape[1])[None, ..., None],
                     axis=(1, 2))

    y_label = jnp.stack([weighted_sum(x, 0, x.shape[-1] // 2),
                         weighted_sum(x, x.shape[-1] // 2, x.shape[-1])],
                        axis=-1) > 0
    y = jnp.where(y_label, 0.5, -0.5)
    return y

  rng_train, rng_test = random.split(random.PRNGKey(1), 2)
  x_train = get_x((n_train, _MAX_SENTENCE_LENGTH, 50), rng_train)
  x_test = get_x((n_test, _MAX_SENTENCE_LENGTH, 50), rng_test)

  y_train, y_test = get_y(x_train), get_y(x_test)
  return x_train, y_train, x_test, y_test


if __name__ == '__main__':
  app.run(main)
