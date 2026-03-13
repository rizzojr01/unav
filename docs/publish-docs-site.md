# Publish Docs Site

## Local Preview

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Open `http://127.0.0.1:8000`.

## Build Static Site

```bash
mkdocs build
```

Output folder:

```text
site/
```

## Deploy to GitHub Pages (Option A: mkdocs gh-deploy)

```bash
mkdocs gh-deploy
```

This pushes rendered docs to `gh-pages` branch.

## Deploy to GitHub Pages (Option B: GitHub Actions)

Use workflow file `.github/workflows/docs.yml` in this repo.

After pushing to `main`:

1. Open repo `Settings -> Pages`
2. Set source to `GitHub Actions`
3. Each push to `main` publishes docs automatically
