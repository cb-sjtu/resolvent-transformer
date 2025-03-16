# testbed

Some class and function names are inherited from `icon-solid-dev`, feel free to change them.

@ Wang Shuo: You can focus on the dataloader part. feel free to create new data class in `src/data/data_utils.py`.

@ Kong Haoming: You can focus on the model and training part. You can start from standard operator learning with classic encoder-decoder transformer. This will also be our baseline.

Check `icon-solid-dev` for more examples.

## environment

```bash
conda env create -f env.yaml # create the environment named sg  
conda activate sg # activate the environment sg
```

## run

```bash
python src/train.py
```

## Custom training
You can create a yaml file in `configs/train_custom.yaml`, and add the following content:

```yaml
defaults:
  - train
  - _self_
# your custom configurations here, for example:
# paths.log_dir: ${paths.root_dir}/scratch/logs/material/
# paths.data_dir: ${paths.root_dir}/scratch/data/material/
```
You can add your own configurations in the file. This file will be ignored by git, so that they are only effective on your machine and won't affect others.
If this file does not exist, `train.yaml` configuration will be used.

## ckpt
To log the checkpoints, please add the callback `ckpt_every_k_steps` in the configs. Currently, the ckpt is saved every `${trainer.val_check_interval}` by default. 

## testing
Testing mode currently under development. I intend to go through all the checkpoints and test them.
Here is the intended usage for testing with multiple validation datasets:
```sh
python src/train.py mode=test \
callbacks.restore_ckpt.ckpt_root="logs/train/runs/${datetime}/checkpoints" \ # required
val_check_interval=${ckpt_interval} # optional
```
Be careful that the `val_check_interval` should be set as the interval of the ckpt. If you set the saving interval as default during training, then you do not need to specify the `val_check_interval` during testing.
