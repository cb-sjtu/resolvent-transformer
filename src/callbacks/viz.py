import lightning as L
import matplotlib.pyplot as plt

from . import viz_utils as vu


class Viz(L.Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0:
            return  # only plot for the first batch

        figs = [[None]]  # a list of list of images (plt.figure or PIL image)
        img = vu.merge_images(figs)  # PIL image
        img.save("test_viz.png")  # for debugging without logging to WandB
        img = vu.fig_to_wandb(img)  # convert to wandb image

        plt.close("all")

        valid_key = list(pl_module.cfg.data.valid.keys())[dataloader_idx]
        valid_name = pl_module.cfg.data.valid[valid_key].name

        for logger in trainer.loggers:
            try:  # noqa: SIM105
                logger.log_image(key=f"{valid_name}", images=[img], step=trainer.global_step)
            except:  # noqa: E722
                pass
