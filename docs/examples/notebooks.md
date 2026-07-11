# Notebooks

Interactive Colab notebooks demonstrating every SafeTune workflow.

## Demos — one per pillar

One notebook per pillar. Each runs to completion with printed output.
Start with **steer_demo** — no GPU needed, ~2 minutes.

**No GPU?** Run 01, 02, 05, 09 (steer, recover, interpret, safety monitoring —
all inference or light patching, no training). **Have a GPU?** Everything
works, including 10 (`full_pipeline`), which is the longest single run since
it exercises all six pillars back to back.

<table class="st-nb-table">
  <thead>
    <tr><th>#</th><th>Notebook</th><th>Pillar</th><th>GPU</th><th>Link</th></tr>
  </thead>
  <tbody>
    <tr><td>01</td><td><code>steer_demo</code> <strong>&larr; start here</strong></td><td>Steer</td><td>No GPU</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1k13VvrAk1NduSH_oJ-3okarnCfsrPYTB" target="_blank">OPEN IN COLAB</a></td></tr>
    <tr><td>02</td><td><code>recover_demo</code></td><td>Recover</td><td>No GPU</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1_w5iaecOTJT8NXzcRZyaM7MzbHK14KvT" target="_blank">OPEN IN COLAB</a></td></tr>
    <tr><td>03</td><td><code>harden_demo</code></td><td>Harden</td><td>GPU helps</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1VGIS5Bk44mCrEqNKIIcbhimSAbFXRQl9" target="_blank">OPEN IN COLAB</a></td></tr>
    <tr><td>04</td><td><code>unlearn_demo</code></td><td>Unlearn</td><td>GPU helps</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1890EkUzDb9q5tNiTCRfI6GjpttTWiEvs" target="_blank">OPEN IN COLAB</a></td></tr>
    <tr><td>05</td><td><code>interpret_demo</code></td><td>Interpret</td><td>No GPU</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1pGKrf8Y9iQcOgR-5SdD3ABVC3cSFWnNT" target="_blank">OPEN IN COLAB</a></td></tr>
    <tr><td>06</td><td><code>evaluate_demo</code></td><td>Evaluate</td><td>GPU helps</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1LSTeRaMaImSLzEatePeZEcWRE2QQvfUl" target="_blank">OPEN IN COLAB</a></td></tr>
  </tbody>
</table>

## Comparisons — pick the right method

Side-by-side runs of multiple methods on the same checkpoint.

<table class="st-nb-table">
  <thead>
    <tr><th>#</th><th>Notebook</th><th>Pillar</th><th>GPU</th><th>Link</th></tr>
  </thead>
  <tbody>
    <tr><td>07</td><td><code>steer_comparison</code></td><td>Steer</td><td>GPU helps</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1QHqYJq-oZvVEGOoIDAI2dztQyppNitzk" target="_blank">OPEN IN COLAB</a></td></tr>
    <tr><td>08</td><td><code>recover_comparison</code></td><td>Recover</td><td>GPU helps</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1o3dV9RhEuyc2wkkm_Vun4Lq0jhka7Max" target="_blank">OPEN IN COLAB</a></td></tr>
  </tbody>
</table>

## Advanced

<table class="st-nb-table">
  <thead>
    <tr><th>#</th><th>Notebook</th><th>Pillar</th><th>GPU</th><th>Link</th></tr>
  </thead>
  <tbody>
    <tr><td>09</td><td><code>safety_monitoring</code></td><td>Evaluate</td><td>No GPU</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1fcfaA-IhEfLpOpjOYdCLtjRDHsc63sA1" target="_blank">OPEN IN COLAB</a></td></tr>
    <tr><td>10</td><td><code>full_pipeline</code></td><td>All pillars</td><td>GPU helps</td><td><a class="st-colab-btn" href="https://colab.research.google.com/drive/1sqWUHknK9reYS3qqEVl75BTFn8DgMADm" target="_blank">OPEN IN COLAB</a></td></tr>
  </tbody>
</table>

## Prefer plain scripts?

The six pillar demos (01–06) and `full_pipeline` (10) each mirror a script in
[`examples/`](https://github.com/Lexsi-Labs/SafeTune/tree/main/examples)
(`examples/quickstart/` or `examples/demos/`); `steer_comparison`,
`recover_comparison`, and `safety_monitoring` (07–09) are notebook-only. Where
a script exists, it's the source of truth; the notebook wraps the same code for
interactive Colab sessions.
