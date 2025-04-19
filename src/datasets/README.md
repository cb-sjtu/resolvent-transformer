# Pytree as Data Structure

We use pytrees as the data structure in this project, due to lightness and flexibility. Pytrees are nested python containers, including lists, tuples, and dictionaries. We don't guarantee the support of other containers.

As a convention for this project, every leaf of pytree is a `torch.Tensor` or `np.ndarray`, with batch size as the leading dimension. This includes the output of `dataset.__getitem__()`, which usually has only one sample (but multiple samples are also supported).

Some utility functions will break the convention, such as `pytree_utils.get_one_batch(pytree, keep_dim=False)`, which is designed for getting one sample from a batch and removing the batch dimension. But these functions are mainly used in callbacks, not in the main training loop.

Use `torch.Tensor`s for data that will be sent to devices — such as model inputs and labels. For other data, like descriptions ([NumPy string arrays](https://numpy.org/devdocs/user/basics.strings.html)), use `np.ndarray`s instead. If certain tensor-like data won't be used on devices, prefer keeping them as `np.ndarray`s to avoid unnecessary device transfer overhead.


# Examples

In this project, the outermost PyTree is typically a dictionary with the following keys:
- **`description`**: A human-readable explanation of the data. Adding descriptions is highly encouraged, as they are helpful for debugging and understanding model behavior. You may also include descriptions in inner containers.
- **`data`**: The input to the model. This should *never* contain any labels. We explicitly pass only `data` into the prediction function.
- **`label`** *(optional)*: The ground truth. You may not need it in tasks like unsupervised or self-supervised learning.

For operator learning, the data can be constructed as:

```python
{
  "description": np.array(['dummy data']*batch_size, dtype=np.dtypes.StringDType()),
  "data": {
    "fx": torch.randn(batch_size, f_len, fx_dim, dtype=torch.float32),
    "fy": torch.randn(batch_size, f_len, fy_dim, dtype=torch.float32),
    "fm": torch.ones(batch_size, f_len, dtype=torch.bool),
    "gx": torch.randn(batch_size, g_len, gx_dim, dtype=torch.float32),
    "gm": torch.ones(batch_size, g_len, dtype=torch.bool),
  },
  "label": torch.randn(batch_size, g_len, gy_dim, dtype=torch.float32),
}
```
Here `f` stands for operator input function, and `g` stands for operator output function. `x` stands for function input, `y` stands for function output, `m` stands for mask. Note that `gy` is not included in `data`, as it's `label` for prediction.

For in-context operator learning with point-wise data, the data can be constructed as:

```python
{
  "description": np.array(['dummy data']*batch_size, dtype=np.dtypes.StringDType()),
  "data": {
    "ex_fx": torch.randn(batch_size, self.ex_num, self.f_len, self.fx_dim, dtype=torch.float32),
    "ex_fy": torch.randn(batch_size, self.ex_num, self.f_len, self.fy_dim, dtype=torch.float32),
    "ex_fm": torch.ones(batch_size, self.ex_num, self.f_len, dtype=torch.bool),
    "ex_gx": torch.randn(batch_size, self.ex_num, self.g_len, self.gx_dim, dtype=torch.float32),
    "ex_gy": torch.randn(batch_size, self.ex_num, self.g_len, self.gy_dim, dtype=torch.float32),
    "ex_gm": torch.ones(batch_size, self.ex_num, self.g_len, dtype=torch.bool),
    "qn_fx": torch.randn(batch_size, self.qn_num, self.f_len, self.fx_dim, dtype=torch.float32),
    "qn_fy": torch.randn(batch_size, self.qn_num, self.f_len, self.fy_dim, dtype=torch.float32),
    "qn_fm": torch.ones(batch_size, self.qn_num, self.f_len, dtype=torch.bool),
    "qn_gx": torch.randn(batch_size, self.qn_num, self.g_len, self.gx_dim, dtype=torch.float32),
    "qn_gm": torch.ones(batch_size, self.qn_num, self.g_len, dtype=torch.bool),
  },
  "label": torch.randn(batch_size, self.qn_num, self.g_len, self.gy_dim, dtype=torch.float32),
}
```
Here `ex` stands for example, and `qn` stands for question. The meaning of `f`, `g`, `x`, `y`, `m` is the same as in the operator learning example. In ICON paper, `f` is termed as `condition`, `q` is termed as `qoi`, `x` is termed as `key`, `y` is termed as `value`. We use shorter names here for convenience. Again, note that `qn_gy` is not included in `data`, as it's `label` for prediction.

The above are just examples. You can make variants as you like.
