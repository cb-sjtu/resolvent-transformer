# Coding Style

This file contains the coding style guidelines for the project. In general, we follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Tensor Operations

**Always prefer einops for tensor operations.**

Benefits: clearer code, fewer bugs, self-documenting dimension names.

Use `import einops` and call functions as `einops.rearrange()`, `einops.reduce()`, `einops.repeat()`, instead of `from einops import rearrange, reduce, repeat`.


```python
# Prefer this
import einops
x = einops.rearrange(x, 'batch seq hidden -> batch hidden seq')

# Instead of this
x = x.transpose(1, 2)
```
