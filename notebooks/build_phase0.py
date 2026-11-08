"""Build notebooks/phase0_spike.ipynb (run with `uv run python notebooks/build_phase0.py`)."""

import nbformat as nbf

cells = [
    nbf.v4.new_markdown_cell(
        "# Phase 0 spike: zero-shot Timestral on NEM prices\n\n"
        "Backtest window: **1–31 Oct 2026**, all 5 regions. Results are produced by "
        "`uv run python -m nemcast.phase0` (see `docs/phase0_results.md`).\n"
        "This notebook scores them and plots examples.\n\n"
        "- **DA-30**: issued 12:45 NEM time on day D, scored on the next trading day\n"
        "  (D+1 04:30 → D+2 04:00, 48 steps).\n"
        "- **ID-5**: issued every 3 h, 5-min resolution, 12 h horizon (144 steps).\n"
        "- DA-30 modes: `uni` (each region alone), `multi` (5 regions as one multivariate series),\n"
        "  `uni_pd` (+ AEMO PD price as a known covariate), `uni_pd_rm` (+ PD price and PD reserve margin).\n"
        "- Baselines: AEMO pre-dispatch (PD), AEMO P5MIN (first hour only), seasonal naive, persistence."
    ),
    nbf.v4.new_code_cell(
        "import json\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n"
        "from nemcast.phase0 import score\n\nR = Path('../data/results')\npd.set_option('display.precision', 1)"
    ),
    nbf.v4.new_markdown_cell("## CPU latency (2 vCPU VM)"),
    nbf.v4.new_code_cell(
        "bench = pd.DataFrame(json.loads((R / 'bench.json').read_text()))\n"
        "bench.pivot_table(index=['case', 'symmetric'], columns='model', values='seconds')"
    ),
    nbf.v4.new_markdown_cell("## DA-30: all models and modes, all regions"),
    nbf.v4.new_code_cell(
        "da = pd.concat([pd.read_parquet(p) for p in sorted(R.glob('da30_*.parquet'))], ignore_index=True)\n"
        "pts = ['aemo_pd', 'naive_day', 'naive_week']\n"
        "score(da, pts, ['model', 'mode'])"
    ),
    nbf.v4.new_markdown_cell("### DA-30 by region (best Timestral config vs baselines)"),
    nbf.v4.new_code_cell(
        "s = score(da, pts, ['model', 'mode'])\nbest = s['timestral_pinball'].idxmin()\nprint('best:', best)\n"
        "best_da = da[(da['model'] == best[0]) & (da['mode'] == best[1])]\n"
        "score(best_da, pts, ['region'])"
    ),
    nbf.v4.new_code_cell(
        "def fan(df, issue, region, point_cols, ax):\n"
        "    d = df[(df['issue'] == issue) & (df['region'] == region)].set_index('valid_time')\n"
        "    ax.fill_between(d.index, d['q05'], d['q95'], alpha=0.15, label='P5–P95')\n"
        "    ax.fill_between(d.index, d['q10'], d['q90'], alpha=0.3, label='P10–P90')\n"
        "    ax.plot(d.index, d['q50'], label='Timestral P50')\n"
        "    for c in point_cols:\n"
        "        ax.plot(d.index, d[c], '--', label=c)\n"
        "    ax.plot(d.index, d['actual'], 'k', lw=1.5, label='actual')\n"
        "    ax.set_title(f'{region} issued {issue:%Y-%m-%d %H:%M}')\n"
        "    ax.set_ylabel('$/MWh')\n\n"
        "issues = sorted(best_da['issue'].unique())\n"
        "fig, axes = plt.subplots(5, 1, figsize=(12, 18))\n"
        "for ax, r in zip(axes, ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']):\n"
        "    fan(best_da, issues[len(issues) // 2], r, ['aemo_pd'], ax)\n"
        "axes[0].legend(ncol=4)\nfig.tight_layout()\nfig.savefig('../docs/img/phase0_da30_fan.png', dpi=80)"
    ),
    nbf.v4.new_markdown_cell("## ID-5: by lead-time bucket"),
    nbf.v4.new_code_cell(
        "ids = pd.concat([pd.read_parquet(p) for p in sorted(R.glob('id5_*.parquet'))], ignore_index=True)\n"
        "ids['lead'] = pd.cut(ids['lead_min'], [0, 60, 180, 360, 720], labels=['0-1h', '1-3h', '3-6h', '6-12h'])\n"
        "pts5 = ['aemo_p5min', 'aemo_pd', 'persistence', 'naive_day']\n"
        "score(ids, pts5, ['model', 'lead'])"
    ),
    nbf.v4.new_code_cell("score(ids[ids['lead'] == '0-1h'], pts5, ['region'])"),
    nbf.v4.new_code_cell(
        "iss = sorted(ids['issue'].unique())\nfig, axes = plt.subplots(2, 1, figsize=(12, 8))\n"
        "for ax, r in zip(axes, ['NSW1', 'SA1']):\n"
        "    fan(ids, iss[len(iss) // 2], r, ['aemo_p5min', 'aemo_pd'], ax)\n"
        "axes[0].legend(ncol=4)\nfig.tight_layout()\nfig.savefig('../docs/img/phase0_id5_fan.png', dpi=80)"
    ),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/phase0_spike.ipynb")
