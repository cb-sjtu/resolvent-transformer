from omegaconf import ListConfig


def decode_materials(materials) -> list[int]:
    if isinstance(materials, int):
        material_ids = list(range(materials))
    elif isinstance(materials, tuple | list | ListConfig) and materials[0] == "20percent":
        material_ids = [i for i in list(range(materials[1])) if i % 5 == 0]
        # maybe use random with controled seed in the future
    elif isinstance(materials, tuple | list | ListConfig) and materials[0] == "80percent":
        material_ids = [i for i in list(range(materials[1])) if i % 5 != 0]
    else:
        raise ValueError(f"Invalid materials: {materials}")
    return material_ids
