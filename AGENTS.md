# Repository guidelines

- Run `pytest -q` before committing to ensure all tests pass.
- Add tests for new features when possible.
- Keep commit messages short and in the present tense.

## Setup

If PyQt or other system libraries are missing, install them and
sync dependencies. Always run these commands before attempting anything else:

```bash
apt-get update
apt-get install -y libegl1 libxslt1.1 libxkbfile1
uv sync
```
