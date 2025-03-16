from pathlib import Path

import torch
from torch.utils.data import Dataset

from .. import data_utils as du
from .utils import decode_materials


class MeshListDataset(Dataset):
    def __init__(self, cfg) -> None:
        """
        try not to use torch.tensor, use numpy instead
        """
        self.cfg = cfg

        self.geometries = list(cfg.geometries)
        material_ids = decode_materials(cfg.materials)
        self.material_ids = material_ids

    def get_filepath(self, root: Path, geometry: str, material_id: int) -> str:
        return root / "cache" / f"Model-{geometry}" / f"data-Model-{geometry}-PolyHyper-{material_id:05d}.npy"

    def get_rawpath(self, root: Path, geometry: str, material_id: int) -> str:
        return root / f"Model-{geometry}" / f"data-Model-{geometry}-PolyHyper-{material_id:05d}"

    def __len__(self):
        return len(self.material_ids)

    def _get_random_element(self, lst: list, num: int):
        # get num of different elements from list
        assert num <= len(lst)
        random_indices = torch.randperm(len(lst))[:num]
        return [lst[i] for i in random_indices]

    def __getitem__(self, idx):
        material_id = self.material_ids[idx]
        if self.cfg.same_geometry:
            geometry_id = self._get_random_element(self.geometries, 1)[0]  # one random geometry
            geometry_snapshot_pool = [
                {"geometry": geometry_id, "snapshot": s}
                for s in range(self.cfg.snapshot_range[0], self.cfg.snapshot_range[1])
            ]
            meshlist_id = self._get_random_element(geometry_snapshot_pool, self.cfg.num_meshes)
        else:
            # loop through all geometries and snapshots to create the pool
            geometry_snapshot_pool = [
                {"geometry": g, "snapshot": s}
                for g in self.geometries
                for s in range(self.cfg.snapshot_range[0], self.cfg.snapshot_range[1])
            ]
            meshlist_id = self._get_random_element(geometry_snapshot_pool, self.cfg.num_meshes)

        mesh_list = du.DataMeshList(
            description=f"DataMeshList, material id: {material_id}, meshlist_id: {meshlist_id}", meshes=[]
        )
        for mesh_id in meshlist_id:
            raw_path = self.get_rawpath(Path(self.cfg.data_folder), mesh_id["geometry"], material_id)
            file_path = self.get_filepath(Path(self.cfg.data_folder), mesh_id["geometry"], material_id)
            this_data = du.DataMesh.load(file_path, raw_path)
            this_data = this_data.get_slice_batch([mesh_id["snapshot"]])
            mesh_list.append(this_data.to_tensor())

        return mesh_list  # DataMeshList
