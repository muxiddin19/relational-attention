# Paper files

This paper was originally submitted to ICDE 2026 and rejected. During
preparation of a revision, a systematic audit found that several headline
claims in that submission did not trace to complete, correctly configured, or
correctly computed experiments (see the paper's Conclusion and Mechanistic
Analysis sections for the full account). The revision corrects each of them
and reports honest null results where that is what the data showed. As of this
commit, the corrected paper is prepared for submission to **Neurocomputing**
(Elsevier); it is not being submitted to ICDE 2027 concurrently (see the
Declaration sections in the manuscript for the current submission status).

| File | Description |
|---|---|
| `relattn_neurocomputing_submission.tex` | Main paper source (elsarticle format) |
| `relattn_neurocomputing_submission.pdf` | Compiled paper |
| `custom.bib` | Bibliography |
| `figures/` | The 6 figures used in the paper (architecture, theory visualization, results, ablations, mechanistic-analysis negative result) |

The experiment results underlying every table and figure are in `../results/`
(see `../results/README.md` for the file-to-table mapping) and
`../analysis/gsm8k_fixed_results/` (for the corrected mechanistic-probing data).
