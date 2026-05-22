# FreeSolv Dashboard

Interactive dashboard exploring the FreeSolv hydration free energy dataset (642 small molecules). Built with Dash + Plotly + RDKit.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open <http://localhost:8050> in a browser.

## Pages

1. **Accuracy Explorer** — calc-vs-expt scatter with live KPIs and filters; click a point to inspect a molecule.
2. **Descriptor Sandbox** — pick X/Y/color from dropdowns; correlation heatmap on the side (click a cell to load that pair).
3. **Confound Revealer** — toggle "Control for |ΔG_expt|" to residualize both axes and watch the apparent chemistry correlation collapse.
4. **Molecule Profile** — search a molecule (or arrive from Page 1's "View full profile") to see its 2D structure, descriptors, expt vs calc, and three most-similar molecules by RDKit fingerprint.

## Deploy to Render

The repo includes `Procfile` and `.python-version`. Render auto-detects them. The default build command (`pip install -r requirements.txt`) and start command (from `Procfile`) work as-is; no overrides needed.
