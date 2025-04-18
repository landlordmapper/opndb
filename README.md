# opndb
Open Property Network DB

Very WIP!!

# Todo:
- [x] init uv environment - should be easy for data scientists to start project and get good environment
- [x] Create directory for data schema v0.1 (pydantic?)
- [ ] ruff formatting / cicd / etc -- only need this when more than one contributor, maybe

- Create gui / webapp repo

# how to run
- install uv on your computer
- `uv sync`

# testing
- `uv run pytest`

# docs

We use sphinx for documentation.

Create new docs files in `docs/source` as `.md` or `.rst` files. Make sure there is a top level `# Header` so the page gets a name.

Build the docs locally by running `cd docs && uv run make html`

We are not hosting the docs yet... coming soon!!!
