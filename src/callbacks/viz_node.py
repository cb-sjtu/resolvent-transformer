import lightning as L

from . import viz_utils as vu


class VizNode(L.Callback):


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0:
            return  # only plot for the first batch

        figs = [[None]] # a list of list of images (plt.figure or PIL image)
        img = vu.merge_images(figs)  # PIL image
        img.save("test_node.png") # just for debugging
        img = vu.fig_to_wandb(img) # convert to wandb image

        for logger in trainer.loggers:
            try:  # noqa: SIM105
                # TODO: replace the key with dataset name
                logger.log_image(key=f"valid_{dataloader_idx}", images=[img], step=trainer.global_step)
            except:  # noqa: E722
                pass
