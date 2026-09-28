# HIV/HCV outbreak in Roka: individual-based model (V3)

Individual-based model (IBM) of the HIV and HCV outbreak in Roka (Cambodia). The model follows every
villager, every syringe and every multidose vial of the practitioner, simulates sexual transmission of
HIV and records the viral genealogy of every infection. Scenarios of introduction (index case,
frequent-patient index case, multidose vials, with or without biological constraints) are compared by
ABC-SMC on the recency of infections (evaluated at each individual's test date) and on the dates of the
HIV genealogy published by Rouet et al. (Clin Infect Dis 2018;66:1733-41).

Selected scenario: frequent-patient index case with multidose vials and biological constraints.
Sensitivity scenarios bound the interval between injections of the index case to at least 3 or 7 days.

Code, data and results: https://github.com/rocheben/roka-hiv-outbreak

## Structure

- `HIV outbreak in Roka_V3.docx`: main text (Emerging Infectious Diseases format).
- `Supplementary Materials_V3.docx`: Supplementary Materials (online Appendix), sections 1 to 9,
  each cited in the main text.
- `data/`: recency counts, population sizes, test dates, published BEAST estimates, env sequences and
  reconstructed trees, ABC settings.
- `code/roka/`: model (`core.py`), scenarios, priors and ABC-SMC (`abc.py`), posterior simulations
  (`posterior.py`), paths, dates, figure style.
- `code/scripts/`: inference (`run_abc.py`, `run_abc_mac.sh`), posterior caches (`compute_posterior.py`),
  one script per figure (`fig1_*.py` to `fig5_*.py`, `figS1_*.py` to `figS3_*.py`), tables and values
  cited in the documents (`make_tables.py`, `parameter_table.py`), observed tree (`build_observed_tree.sh`),
  whole pipeline (`make_all.sh`).
- `code/manuscript/`: generation of the two documents (Node.js, package `docx`).
- `figures/`: PNG figures, and `figures/eid/` the files for submission (PDF and TIFF 600 dpi).
- `resultats/`: ABC results (`abc/`), posterior caches (`cache/`), tables and `numbers.json`.
- `old/`: files not used by the article.

All paths are relative to the project root: the scripts can be run from any directory.

## Requirements

- Python ≥ 3.10 with the packages of `code/requirements.txt` (numpy ≥ 2 is required to read the
  ABC results in `resultats/abc/*.pkl`). `run_abc_mac.sh` creates this environment in `~/.venvs/roka`
  on first use; otherwise `python3 -m venv ~/.venvs/roka && ~/.venvs/roka/bin/pip install -r code/requirements.txt`.
- Node.js ≥ 18, only to rebuild the two .docx documents (`cd code/manuscript && npm install` once).
- FastTree and TreeTime, only to rebuild the observed tree (`build_observed_tree.sh`).
- The shell scripts need bash (macOS, Linux, or WSL/Git Bash on Windows); the Python scripts run
  anywhere. The numba compilation cache is kept in `~/.cache/roka_numba`, outside the project folder.

## Running (Mac Terminal)

```
cd "<project folder>"
bash code/scripts/run_abc_mac.sh            # ABC, all scenarios (resumes automatically)
bash code/scripts/run_abc_mac.sh 9 11       # selected scenarios only
cd code/manuscript && npm install && cd ../..   # once, for the documents
bash code/scripts/make_all.sh               # caches, figures, tables and both documents
python3 code/scripts/fig2_fit.py            # a single figure
```
