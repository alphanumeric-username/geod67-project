# FWI with Unets

This project is a reproduction of the paper: [Seismic full-waveform inversion regularized with a migration image](https://doi.org/10.1190/geo2023-0419.1)

## Installation

<!-- **WARNING! This project must be run on a Linux environment**: this is due to its dependence to the Devito library, which only works on Linux based OSes. -->

- Using conda:
    ```
    conda env create -f environment.yml
    conda activate geod67-unet-fwi
    ```

- Using pip:

    It is recommended to create a virtual environment to install the required packages, which can be done with by executing the follwing line,
    ```
    pip install -r requirements.txt
    ```