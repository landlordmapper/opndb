# opndb
Open Property Network DB

# Installation:
- Install `uv` if not already installed (Python package manager)
- Pull down repository into Jupyter-enabled IDE
- Create environment and install packages
  * `cd opndb` - `cd` into opndb root directory
  * `uv venv` - Create `uv` virtual environment
  * `source .venv/bin/activate` - Activate `uv` virtual environment
  * `uv pip install .` - Installs all packages & dependencies specified in pyproject.toml
  * [OPTIONAL] `uv pip install pip` - This may be required to enable jupyter notebooks in some IDEs
- Configure local jupyter server to use the `uv` environment

