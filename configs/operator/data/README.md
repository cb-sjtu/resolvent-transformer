This folder contains the data configs.

`train` will be a dictionary of data configs. In lightning datamodule, we will cycle through the these datasets.

`valid` will be a dictionary of data configs. In lightning datamodule, we will create a list of dataloopers for separate validation.

Keep `name`, it will be used in logging. Keep `description`, it will explain the data.