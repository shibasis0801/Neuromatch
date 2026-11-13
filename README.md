# Dynamical neural change in visual cortex during learning

My group project from the Neuromatch Academy course in Computational Neuroscience (July 2026). We asked how representations in mouse visual cortex change trial by trial within a single session, and whether reward changes those dynamics.

## Question

Zhong et al. (2025) recorded about 70,000 neurons at once across mouse visual cortex while mice ran through two virtual corridors, one with leaf and one with circle wall textures. Supervised mice got a sugar-water reward in the leaf corridor. Unsupervised mice saw the same corridors with no reward. Most of the change in visual representations happened in both groups, so it did not need reward.

Their comparison was between sessions, before and after learning. We looked inside sessions. For each neuron, d′ measures how well its activity separates leaf from circle. We followed the distribution of d′ across the population as a session goes on:

| | Statistic | Hypothesis |
|---|---|---|
| H1 | Median d′ | The median moves differently with and without reward |
| H2 | Standard deviation of d′ | The distribution widens faster early in the first supervised session |
| H3 | Fraction of selective neurons (\|d′\| ≥ 0.3) | Recruitment rises steeply early in the first supervised session and stays flat without reward |

## My part

- **Data pipeline**, `code/data.py`. It pulls the release from the Figshare API, checks every file against its published checksum, and loads sessions, frames, trials and neurons into DuckDB, so any question about the experiment is one SQL query. Neural activity stays in the release's compressed SVD form and is rebuilt on demand for any block of neurons and frames.
- **Within-session d′ analysis.** `notebooks/dprime-original.ipynb` is the notebook I wrote during the course. `notebooks/dprime.ipynb` is the same analysis rebuilt on the data pipeline and plotting library, for all 26 sessions and each visual area.

## Method

- 13 mice from the first texture experiment (4 supervised, 9 unsupervised), each imaged in one session before and one after learning: 26 sessions.
- Each neuron's activity is reconstructed from the 400-component SVD in the release.
- A trial is one pass through a corridor's texture zone. Only frames where the mouse is moving count, and a trial needs at least 3 of them. The i-th leaf trial is paired with the i-th circle trial.
- In each window of 20 trial pairs, sliding one pair at a time, every neuron gets d′ = 2(μ_leaf − μ_circle) / (σ_leaf + σ_circle).
- In each window I summarise the population: median, standard deviation and skewness of d′, the fraction of leaf-selective (d′ ≥ 0.3) and circle-selective (d′ ≤ −0.3) neurons, and running speed. I do this for all neurons and separately for V1 and the medial, lateral and anterior higher visual areas.
- As a group we then fitted generalized additive mixed models (GAMMs) to these trajectories. As a bonus, we trained ridge models on the unsupervised mice to predict supervised mice's after-learning statistics from their before-learning state, with leave-one-mouse-out validation.

## Results

From our final presentation (`slides.pdf`):

- **Running speed matters.** It differs between mice and changes over a session, and it modulates visual responses, so it is a confound to control.
- **Before learning: similar beginnings, divergent paths.** Median d′ overlapped between groups. The standard deviation differed (p = .014), and was higher in unsupervised mice. Unsupervised mice pulled ahead in leaf and circle selectivity later in the session.
- **After learning: reward yields preference.** Only the fraction of leaf-selective neurons differed between groups, which points to plasticity specific to the rewarded stimulus. Median d′ and circle selectivity were stable from before to after learning.
- **Prediction (bonus).** In the higher visual areas the model beat a persistence baseline (mean absolute error 0.249 vs 0.439 d′ for medial and anterior areas together). In V1 it did not (0.523 vs 0.406).

`dprime.ipynb` reproduces the d′ distributions and trajectories straight from the public release. The GAMM statistics and the prediction model are in the slides.

## Repository

```
code/data.py                     Figshare API to DuckDB
code/plot.py                     figures for the notebooks
notebooks/data.ipynb             the dataset, explained
notebooks/dprime.ipynb           the d′ analysis on data.py and plot.py
notebooks/dprime-original.ipynb  my course notebook, unchanged
literature/                      Zhong et al. (2025)
slides.pdf                       our final presentation
```

## Running it

Python 3.11 or newer:

```bash
pip install duckdb jupyterlab matplotlib numpy pandas requests scikit-learn scipy
cd notebooks
jupyter lab
```

Run `data.ipynb` first. The first run downloads about 4.3 GiB to `~/zhong2025`: four behaviour files, plus the SVD and cortical map of each of the 26 recordings. Set `ZHONG_DATA` to keep the data somewhere else. After that, `dprime.ipynb` runs in about a minute on a laptop.

## Team

Group Ratatouille, pod Tokoloshe Black Cumin: Grace Antony, Aditi Kailkhura, Saloni Komal, Tomoya Kubota, Mira Malekon, Shibasis Patnaik, Tanya Sharma, Maria Shaukat, Janviere Yau and Rui Zheng. Teaching assistants: Lina Alqam and Amalya Hakobyan. The sliding-window d′ follows the definition Rui Zheng set out for the group.

## Data and references

Data: Zhong et al. (2025), Janelia Figshare, [10.25378/janelia.28811129.v2](https://doi.org/10.25378/janelia.28811129.v2), CC BY 4.0.

- Zhong, L., Baptista, S., Gattoni, R., et al. (2025). Unsupervised pretraining in biological neural networks. *Nature, 644*, 741–748. [10.1038/s41586-025-09180-y](https://doi.org/10.1038/s41586-025-09180-y) (PDF in `literature/`, CC BY 4.0)
- Henschke, J. U., Dylda, E., Katsanevaki, D., et al. (2020). Reward association enhances stimulus-specific representations in primary visual cortex. *Current Biology, 30*(10), 1866–1880. [10.1016/j.cub.2020.03.018](https://doi.org/10.1016/j.cub.2020.03.018)
- Jurjut, O., Georgieva, P., Busse, L., & Katzner, S. (2017). Learning enhances sensory processing in mouse V1 before improving behavior. *Journal of Neuroscience, 37*(27), 6460–6474. [10.1523/JNEUROSCI.3485-16.2017](https://doi.org/10.1523/JNEUROSCI.3485-16.2017)
- Deitch, D., Rubin, A., & Ziv, Y. (2021). Representational drift in the mouse visual cortex. *Current Biology, 31*(19), 4327–4339. [10.1016/j.cub.2021.07.062](https://doi.org/10.1016/j.cub.2021.07.062)
- Niell, C. M., & Stryker, M. P. (2010). Modulation of visual responses by behavioral state in mouse visual cortex. *Neuron, 65*(4), 472–479. [10.1016/j.neuron.2010.01.033](https://doi.org/10.1016/j.neuron.2010.01.033)
