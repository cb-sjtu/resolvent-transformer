
## Return of validation_step and test_step

`validation_step` and `test_step` should return a pytree containing these top-level keys:
- `preds`: A pytree containing the model's predictions
- `errors`: A pytree containing detailed prediction errors (e.g., image format errors)
- `metrics`: A pytree containing evaluation metrics. It is strongly suggested that each metric maintain the batch dimension with shape `(batch_size, xxx)`, for example, `(batch_size, 2, 2)`. While `xxx` can have multiple dimensions, the flattened size should remain reasonably small.
