## Dataset Information

- **Source**: Kuramoto-Shivashinsky (KS) equation simulations
- **Original Authors**: Brandstetter, Johannes and Welling, Max and Worrall, Daniel E
- **Repository**: [LPSDA GitHub Repository](https://github.com/brandstetter-johannes/LPSDA)
- **License**: MIT License

## Data Structure

```yaml
file_structure:
      data_location: "data directory containing three HDF5 files: KS_train_512.h5, KS_valid_512.h5, KS_test_512.h5"
      top_level_groups: "Each file contains a top-level group named after the split ('train', 'valid', 'test')"
      datasets_per_group: "Five datasets with consistent naming but different dimensions"
    
    datasets:
      pde_140-256:
        description: "KS equation solution tensor"
        shape: "512*140*256"
        dimensions: "samples*time_steps*spatial_points"
        details: "512 independent samples, each retaining the last 140 time steps (nt_effective=140), with 256 spatial discretization points (nx=256)"
      
      x:
        description: "Spatial coordinates for each sample"
        shape: "512*256" 
        dimensions: "samples*spatial_points"
        details: "Spatial coordinates from 0 to L(1-1/256)=63.75, with step size L/nx=0.25"
      
      dx:
        description: "Spatial grid spacing"
        shape: "512"
        dimensions: "samples"
        details: "Each element equals 0.25, representing the spatial step size"
      
      t:
        description: "Temporal coordinates"
        shape: "512*140"
        dimensions: "samples*time_steps"
        details: "Physical time moments for the last 140 time points of each trajectory. Time sampling: 500 equally-spaced points in [0,T] then truncated to last 140 steps, where T randomly varies between 90 and 110"
      
      dt:
        description: "Temporal step size"
        shape: "512"
        dimensions: "samples"
        details: "Time step size T/(nt-1)=T/499 for each sample, varies slightly between samples due to random T"
```

## License Information

The KS dataset is released under the following MIT License:

```text
MIT License

Copyright (c) 2023 brandstetter-johannes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
