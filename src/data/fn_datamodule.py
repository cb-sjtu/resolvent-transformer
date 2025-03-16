import lightning as L
import torch

# typing
from omegaconf import DictConfig

from . import data_utils as du
from .dataloader import CycleDataLooper, DataLooper
from .datasets import EquationDataset, MeshListDataset


class FnLitDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # <- call the data generation script here
        pass

    def setup(self, stage: str = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

    def dataloader_eqn_mix_geometry(self, cfg):
        """
        geometry will be mixed in one batch.
        return: datalooper.
        """

        def collate_fn(data_list: list[du.DataEqn]):
            data = du.concat_data(data_list)
            if cfg.eqns_per_prompt is not None:
                # slice the data to get a random number of equations
                num_eqns = torch.randint(cfg.eqns_per_prompt[0], cfg.eqns_per_prompt[1] + 1, (1,)).item()
                data = data.get_slice_eqn(list(range(num_eqns)))
            return data

        batch_size = cfg.batch_size_per_process // cfg.batch_size_per_record
        trainset = EquationDataset(
            cfg=cfg, geometries=cfg.geometries, materials=cfg.materials, trainer=self.trainer
        )  # use trainer to get info like rank, steps
        return DataLooper(trainset, cfg, batch_size, collate_fn)

    def dataloader_eqn_cycle_geometry(self, cfg):
        """
        We will cycle over the geometries, i.e. one batch share the same geometry.
        This is essentially for snapshot split without slicing,
        since different geometries have different number of nodes/equations.
        return: datalooper.
        """

        def collate_fn(data_list: list[du.DataEqn]):
            data = du.concat_data(data_list)
            if cfg.eqns_per_prompt is not None:
                # slice the data to get a random number of equations
                num_eqns = torch.randint(cfg.eqns_per_prompt[0], cfg.eqns_per_prompt[1] + 1, (1,)).item()
                data = data.get_slice_eqn(list(range(num_eqns)))
            return data

        batch_size = cfg.batch_size_per_process // cfg.batch_size_per_record
        dataloopers = []
        for geometry in cfg.geometries:
            trainset = EquationDataset(
                cfg=cfg, geometries=[geometry], materials=cfg.materials, trainer=self.trainer
            )  # use trainer to get info like rank, steps
            datalooper = DataLooper(trainset, cfg, batch_size, collate_fn)
            dataloopers.append(datalooper)
        return CycleDataLooper(dataloopers)


    def dataloader_mesh_list(self, cfg):
        """
        a MeshList with multiple meshes.
        :return: The validation dataloader, will return a mesh list, with no batching.
        """
        batch_size = None  # we will not batch the mesh list
        validset = MeshListDataset(cfg=cfg)
        datalooper = DataLooper(validset, cfg, batch_size, collate_fn=None)
        return datalooper


    def train_dataloader(self):
        # cycle through the dataloaders of the different splits
        dataloopers = []
        for i, (key, cfg) in enumerate(self.cfg.data.train.items()):
            print(f"train dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
            if cfg.geometry == "mix":
                dataloopers.append(self.dataloader_eqn_mix_geometry(cfg))
            elif cfg.geometry == "cycle":
                dataloopers.append(self.dataloader_eqn_cycle_geometry(cfg))
            else:
                raise ValueError(f"Invalid geometry: {cfg.geometry}")
        return CycleDataLooper(dataloopers) # return a single cycle dataloader

    def val_dataloader(self):
        """Create and return the validation dataloader.
        :return: a list of dataloopers for validation
        """
        dataloopers = []
        for i, (key, cfg) in enumerate(self.cfg.data.valid.items()):
            print(f"valid dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
            if cfg.task == "mesh_list":
                dataloopers.append(self.dataloader_mesh_list(cfg))
            else:
                raise ValueError(f"Invalid task: {cfg.task}")
        return dataloopers # return a list of dataloopers for separate validation

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        pass

    def teardown(self, stage: str = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass
