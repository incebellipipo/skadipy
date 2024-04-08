[![Python package](https://github.com/incebellipipo/skadipy/actions/workflows/run_test.yml/badge.svg)](https://github.com/incebellipipo/skadipy/actions/workflows/run_test.yml)

# mccontrolpy

A python library for solving the control allocation problem for marine craft with different types of actuators and control allocation methods.

## Installation



1. Clone the package and create a virtual environment.
    ```bash
    python -m venv venv
    ```

2. Activate that virtual environment. It would change depending on the OS.
    On Linux or Mac
    ```bash
    source venv/bin/activate
    ```
    or on Windows
    ```powershell
    venv\Scripts\activate
    ```

3. Install package dependencies
    ```bash
    pip install -r requirements.txt
    ```

    Install packages if you wish to run notebooks
    ```bash
    pip install -r requirements.examples.txt
    ```
    From now on you should be able to run the notebooks and see the results.

4. Install the package
    ```bash
    pip install -e .
    ```

5. Test the installation
    ```bash
    python -c "import skadipy; print(skadipy.__version__)"
    ```

    If it doesn't throw any error, the installation was successful.

## Directory Structure

- `src`: Source code for the library
- `tests`: Unit tests for the library
- `examples`: Examples of how to use the library
- `notebooks`: Jupyter notebooks for the library showing examples and documentation.
- `docs`: Documentation for the library
- `bin`: Scripts for running the library

## Acknowledgements

Work funded by the Research Council of Norway (RCN) through NTNU AMOS (RCN project 223254), SFI AutoShip (RCN project 309230), and the Polish National Centre for Research and Development through the ENDURE project (NOR/POLNOR/ENDURE/0019/2019-00).