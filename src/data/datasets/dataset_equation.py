import glob
import os
from pathlib import Path

import h5py
import lightning as L
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from .. import data_utils as du
from .utils import decode_materials


class EquationDataset(Dataset):
    def __init__(
        self, cfg: DictConfig, geometries: list[str], materials: tuple | list | ListConfig | int, trainer: L.Trainer
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.rank = trainer.global_rank
        self.world_size = trainer.world_size

        if cfg.split == "material":
            cache_folder = self.get_cache_folder(Path(cfg.data_folder), "material", geometries, materials)
            eqn_filepaths = glob.glob(str(cache_folder / "*.h5"))
            if len(eqn_filepaths) == 0:
                self.cache_eqn_data_material(
                    root=Path(cfg.data_folder), geometries=geometries, materials=materials, cache_folder=cache_folder
                )
                eqn_filepaths = glob.glob(str(cache_folder / "*.h5"))  # check again
                if len(eqn_filepaths) == 0:
                    raise ValueError(f"Failed to cache equation data for {cache_folder}")

        elif cfg.split == "geometry":
            eqn_filepaths = []
            for geometry in geometries:
                # use material since we split geometry with for loop
                cache_folder = self.get_cache_folder(Path(cfg.data_folder), "material", [geometry], materials)
                this_eqn_filepaths = glob.glob(str(cache_folder / "*.h5"))
                if len(this_eqn_filepaths) == 0:
                    self.cache_eqn_data_material(
                        root=Path(cfg.data_folder),
                        geometries=[geometry],
                        materials=materials,
                        cache_folder=cache_folder,
                    )
                    this_eqn_filepaths = glob.glob(str(cache_folder / "*.h5"))
                    if len(this_eqn_filepaths) == 0:
                        raise ValueError(f"Failed to cache equation data for {cache_folder}")
                eqn_filepaths.extend(this_eqn_filepaths)

        elif cfg.split == "snapshot":
            eqn_filepaths = []
            for geometry in geometries:
                cache_folder = self.get_cache_folder(Path(cfg.data_folder), "snapshot", [geometry], materials)
                this_eqn_filepaths = glob.glob(str(cache_folder / "*.h5"))
                if len(this_eqn_filepaths) == 0:
                    self.cache_eqn_data_snapshot(
                        root=Path(cfg.data_folder), geometry=geometry, materials=materials, cache_folder=cache_folder
                    )
                    this_eqn_filepaths = glob.glob(str(cache_folder / "*.h5"))  # check again
                    if len(this_eqn_filepaths) == 0:
                        raise ValueError(f"Failed to cache equation data for {cache_folder}")
                eqn_filepaths.extend(this_eqn_filepaths)

        else:
            raise ValueError(f"Invalid split: {cfg.split}")

        self.indices = []
        for file_path in eqn_filepaths:
            with h5py.File(file_path, "r") as f:
                self.indices.extend([(file_path, key) for key in f])

        if self.rank == 0:
            print(f"{cfg.name} eqn_filepaths:")
            for file_path in eqn_filepaths:
                print(file_path)
            print(f"{cfg.name} total number of records: {len(self.indices)}")

        # deprecated, but keep it as comment for now
        # self.compute_statistics()

    def get_cache_folder(self, root: Path, split: str, geometries: list[str], materials: list | int) -> str:
        if isinstance(materials, int):
            materials = [materials]
        cache_folder = f"{split}-{''.join(geometries)}-{''.join([str(m) for m in materials])}"
        cache_folder = root / "cache_eqn" / cache_folder
        return cache_folder

    def get_mesh_filepath(self, root: Path, geometry: str, material_id: int) -> str:
        return root / "cache" / f"Model-{geometry}" / f"data-Model-{geometry}-PolyHyper-{material_id:05d}.npy"

    def get_raw_filepath(self, root: Path, geometry: str, material_id: int) -> str:
        return root / f"Model-{geometry}" / f"data-Model-{geometry}-PolyHyper-{material_id:05d}"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_path, group_name = self.indices[idx]
        with h5py.File(file_path, "r") as f:
            group = f[group_name]
            A = torch.tensor(group["A"][:], dtype=torch.float32)  # (N, 8, 2, 2)
            XI = torch.tensor(group["XI"][:], dtype=torch.float32)  # (N, 8, 2)
            mask = torch.tensor(group["mask"][:], dtype=torch.bool)  # (N, 8)
        if self.cfg.eqns_per_prompt is not None:
            # random sample a subset of eqns with largest possible number, will be sliced together in collate_fn
            assert self.cfg.eqns_per_prompt[1] * self.cfg.batch_size_per_record <= A.shape[0]
            eqn_ids = torch.randperm(A.shape[0])[: self.cfg.eqns_per_prompt[1] * self.cfg.batch_size_per_record]
            A = A[eqn_ids].view(self.cfg.batch_size_per_record, -1, *A.shape[1:])
            XI = XI[eqn_ids].view(self.cfg.batch_size_per_record, -1, *XI.shape[1:])
            mask = mask[eqn_ids].view(self.cfg.batch_size_per_record, -1, *mask.shape[1:])
        else:  # no slicing
            assert self.cfg.batch_size_per_record == 1
            A = A[None, ...]  # add batch dimension, (1, n, 8, 2, 2)
            XI = XI[None, ...]  # (1, n, 8, 2)
            mask = mask[None, ...]  # (1, n, 8)

        eqn_data = du.DataEqn(
            description=[f"{file_path}_{group_name}_eqns_{self.cfg.eqns_per_prompt}"] * A.shape[0],
            A=A,
            XI=XI,
            mask=mask,
        )

        if self.cfg.rotate == "no":
            angle = None
        elif self.cfg.rotate == "full":
            angle = torch.rand(*eqn_data.A.shape[:2]) * 2 * torch.pi  # (batch_size, n_eqns)
        elif self.cfg.rotate == "half":
            angle = torch.rand(*eqn_data.A.shape[:2]) * 2 * torch.pi  # (batch_size, n_eqns)
            angle_mask = torch.rand(*eqn_data.A.shape[:1]) < 0.5  # (batch_size,)
            angle[angle_mask, :] = 0  # set half batch to no rotation
        else:
            raise ValueError(f"Invalid rotate: {self.cfg.rotate}")
        eqn_data = self.rotate(eqn_data, angle)

        if self.cfg.flip == "no":
            flip = None
        elif self.cfg.flip == "full":
            flip = torch.sign(torch.rand(*eqn_data.A.shape[:2]) * 2 - 1.0)  # (batch_size, n_eqns)
        elif self.cfg.flip == "half":
            flip = torch.sign(torch.rand(*eqn_data.A.shape[:2]) * 2 - 1.0)  # (batch_size, n_eqns)
            flip_mask = torch.rand(*eqn_data.A.shape[:1]) < 0.5  # (batch_size,)
            flip[flip_mask, :] = 1  # set half batch to no flip
        else:
            raise ValueError(f"Invalid flip: {self.cfg.flip}")
        eqn_data = self.flip(eqn_data, flip)

        return eqn_data

    def cache_eqn_data_material(
        self, root: Path, geometries: list[str], materials: tuple | list | ListConfig | int, cache_folder: Path
    ):
        """
        each record represent a material. Different geometries and snapshots are all mixed together
        so each record has num_geometries * num_snapshots * num_free_nodes equations.
        when geometries is a list of one element, then each record has num_snapshots * num_free_nodes equations.
        we use this function to generate the dataset for:
        1, material split.
        2, geometry split, with geometries being a list of one element.
        """
        print(f"caching equation data for {cache_folder}")
        if self.rank == 0 and not cache_folder.exists():
            os.makedirs(cache_folder)
        torch.distributed.barrier()  # make sure the folder is created

        total_material_ids = decode_materials(materials)
        total_num_files = 8 * self.world_size
        file_ids = [i for i in range(total_num_files) if i % self.world_size == self.rank]
        material_ids_split = [total_material_ids[i::total_num_files] for i in file_ids]
        for file_id, material_ids in zip(file_ids, material_ids_split, strict=False):
            with h5py.File(cache_folder / f"data-eqn-{file_id}.h5", "w") as h5_file:
                for material_id in material_ids:
                    group = h5_file.create_group(f"material_id_{material_id}")
                    data_cache = []
                    for geometry in geometries:
                        raw_path = self.get_raw_filepath(root, geometry, material_id)
                        mesh_path = self.get_mesh_filepath(root, geometry, material_id)
                        try:
                            this_data = du.DataMesh.load(mesh_path, raw_path)
                        except Exception as e:
                            print(f"Error loading file {raw_path}, {mesh_path}: {e}")
                            continue
                        this_data = this_data.get_slice_batch(
                            list(range(1, len(this_data.X)))
                        )  # drop the first time step
                        A = du.slice_nodes_set(this_data.A, this_data.FREEset)  # (N_t, nfree, 8, 2, 2)
                        XI = du.slice_nodes_set(this_data.XI, this_data.FREEset)  # (N_t, nfree, 8, 2)
                        mask = du.slice_nodes_set(this_data.mask, this_data.FREEset)  # (N_t, nfree, 8)
                        this_data_eqn = du.DataEqn(
                            A=rearrange(A, "t e ... -> (t e) ..."),  # (N_t * nfree, 8, 2, 2)
                            XI=rearrange(XI, "t e ... -> (t e) ..."),  # (N_t * nfree, 8, 2)
                            mask=rearrange(mask, "t e ... -> (t e) ..."),
                        )  # (N_t * nfree, 8)
                        data_cache.append(this_data_eqn)
                    data_cache = du.stack_data(data_cache)
                    group.create_dataset("A", data=data_cache.A)
                    group.create_dataset("XI", data=data_cache.XI)
                    group.create_dataset("mask", data=data_cache.mask)

        torch.distributed.barrier()  # wait for all processes to finish caching

    def cache_eqn_data_snapshot(
        self, root: Path, geometry: str, materials: tuple | list | ListConfig | int, cache_folder: Path
    ):
        """
        each record represent a snapshot,
        each record has num_free_nodes equations.
        """
        print(f"caching equation data for {cache_folder}")
        if self.rank == 0 and not cache_folder.exists():
            os.makedirs(cache_folder)
        torch.distributed.barrier()  # make sure the folder is created
        total_material_ids = decode_materials(materials)

        total_num_files = 8 * self.world_size
        file_ids = [i for i in range(total_num_files) if i % self.world_size == self.rank]
        material_ids_split = [total_material_ids[i::total_num_files] for i in file_ids]

        for file_id, material_ids in zip(file_ids, material_ids_split, strict=False):
            with h5py.File(cache_folder / f"data-eqn-{file_id}.h5", "w") as h5_file:
                for material_id in material_ids:
                    raw_path = self.get_raw_filepath(root, geometry, material_id)
                    mesh_path = self.get_mesh_filepath(root, geometry, material_id)
                    try:
                        this_data = du.DataMesh.load(mesh_path, raw_path)
                    except Exception as e:
                        print(f"Error loading file {raw_path}, {mesh_path}: {e}")
                        continue
                    this_data = this_data.get_slice_batch(list(range(1, len(this_data.X))))  # drop the first time step
                    A = du.slice_nodes_set(this_data.A, this_data.FREEset)  # (N_t, nfree, 8, 2, 2)
                    XI = du.slice_nodes_set(this_data.XI, this_data.FREEset)  # (N_t, nfree, 8, 2)
                    mask = du.slice_nodes_set(this_data.mask, this_data.FREEset)  # (N_t, nfree, 8)
                    for t in range(A.shape[0]):
                        group = h5_file.create_group(f"material_id_{material_id}_t_{t}")
                        group.create_dataset("A", data=A[t])  # (nfree, 8, 2, 2)
                        group.create_dataset("XI", data=XI[t])  # (nfree, 8, 2)
                        group.create_dataset("mask", data=mask[t])  # (nfree, 8)

        torch.distributed.barrier()  # wait for all processes to finish caching

    def rotate(self, data: du.DataEqn, angle: torch.Tensor | None = None) -> du.DataEqn:
        if angle is None:
            return data  # no rotation
        mat = torch.stack([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], dim=-1)
        mat = mat.view(*data.A.shape[:2], 1, 2, 2)  # (batch_size, n_eqns, 1, 2, 2)
        data.A = mat @ data.A  # (batch_size, n_eqns, 8, 2, 2)
        return data

    def flip(self, data: du.DataEqn, flip: torch.Tensor | None = None) -> du.DataEqn:
        if flip is None:
            return data  # no flip
        mat = torch.stack([torch.ones_like(flip), torch.zeros_like(flip), torch.zeros_like(flip), flip], dim=-1)
        mat = mat.view(*data.A.shape[:2], 1, 2, 2)  # (batch_size, n_eqns, 1, 2, 2)
        data.A = mat @ data.A  # (batch_size, n_eqns, 8, 2, 2)
        return data

    # def compute_statistics(self):
    #     # loop for the whole dataset to compute statistics for normalization
    #     A_list = []
    #     XI_list = []

    #     for idx in range(len(self)):
    #         eqn_data = self[idx]

    #         A = eqn_data.A  # (1, n, 8, 2, 2)
    #         XI = eqn_data.XI  # (1, n, 8, 2)
    #         mask = eqn_data.mask  # (1, n, 8)

    #         valid_A = torch.masked_select(A, mask[..., None, None]).view(-1, *A.shape[-2:])  # (n * 8, 2, 2)
    #         valid_XI = torch.masked_select(XI, mask[..., None]).view(-1, *XI.shape[-1:])  # (n * 8, 2)
    #         A_list.append(valid_A)
    #         XI_list.append(valid_XI)
    #         # mask = eqn_data.mask

    #     # compute statistics for A, XI
    #     As = torch.cat(A_list, dim=0)
    #     XIs = torch.cat(XI_list, dim=0)
    #     A_std, A_mean = torch.std_mean(As, dim=0)
    #     XIs_std, XIs_mean = torch.std_mean(XIs, dim=0)

    #     self.A_std = A_std
    #     self.A_mean = A_mean
    #     self.XIs_std = XIs_std
    #     self.XIs_mean = XIs_mean

    #     print('=' * 1000)
    #     print(f"A_std: {A_std}, A_mean: {A_mean}")
    #     print(f"XIs_std: {XIs_std}, XIs_mean: {XIs_mean}")
