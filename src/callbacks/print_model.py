import lightning as L
from tabulate import tabulate

# typing
from src.utils.rich_utils import print_config_tree


class PrintModel(L.Callback):
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # print the model details customly
        pl_module.print(type(pl_module.net))
        model = pl_module.net.module if hasattr(pl_module.net, "module") else pl_module.net
        headers = ["Parameter Name", "Shape", "Requires Grad"]
        table_data = [(name, str(param.shape), param.requires_grad) for name, param in model.named_parameters()]
        pl_module.print(tabulate(table_data, headers=headers, tablefmt="grid"))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pl_module.print(f"Total Parameters: {total_params:,}")
        pl_module.print(f"Trainable Parameters: {trainable_params:,}")

        pl_module.print(f"SDPA backends: {pl_module.sdpa_backends}")
        print_config_tree(pl_module.cfg, resolve=True)

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx):
        cfg = pl_module.cfg
        data, label = batch["data"], batch["label"]
        if batch_idx < 20:
            pl_module.print("=" * 40 + f"Train Data # {batch_idx}" + "=" * 40)
            pl_module.print("Input data", type(data))
            pl_module.print(data.get_print_info(print_lv=cfg.print_lv))
            pl_module.print("Label data", type(label))
            pl_module.print(label.get_print_info(print_lv=cfg.print_lv))
            pl_module.print("=" * 100)

    def on_validation_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx, dataloader_idx=0
    ):
        cfg = pl_module.cfg
        data, label = batch["data"], batch["label"]
        if batch_idx < 20 and trainer.global_step < 100:
            pl_module.print("=" * 40 + f"Valid Dataset # {dataloader_idx} - Batch {batch_idx}" + "=" * 40)
            pl_module.print("Input data", type(data))
            pl_module.print(data.get_print_info(print_lv=cfg.print_lv))
            pl_module.print("Label data", type(label))
            pl_module.print(label.get_print_info(print_lv=cfg.print_lv))
            pl_module.print("=" * 100)
