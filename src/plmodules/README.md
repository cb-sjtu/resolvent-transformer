# Return of validation_step

Note that the value returned by `validation_step` is passed directly as the `outputs` argument to `on_validation_batch_end` in all callbacks.

`validation_step` in plmodules should return a pytree. It is strongly suggested that each leaf is a `torch.Tensor` or `np.ndarray`. Keep the leading dimension as batch size if possible. In other words, do not pool over batch dimension.

It is suggested that the returned pytree contains these top-level keys:

- `preds`: A pytree containing the model's detailed predictions, e.g., in the format of images
- `errors`: A pytree containing detailed prediction errors, e.g., in the format of images
- `metrics`: A pytree containing evaluation metrics. It is strongly suggested that each metric has a shape of `(batch_size, ...)`, for example, `(batch_size, 2, 2)`, where `...` can have multiple dimensions, but the flattened size should remain reasonably small.
