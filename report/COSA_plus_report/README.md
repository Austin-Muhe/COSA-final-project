# COSA+ Report

Editable LaTeX source:

```bash
/home/wwww/projects/COSA-final-project/report/COSA_plus_report/main.tex
```

The report follows the CPS3830 required structure: title, abstract, introduction/problem definition, background, methodology, implementation, experiments, results/analysis, conclusion, references, AI usage statement, and team contribution statement.

Before submission, replace all `Name / Student ID` placeholders and fill in the real team contribution statement.

To compile locally from this folder:

```bash
cd /home/wwww/projects/COSA-final-project/report/COSA_plus_report
pdflatex main.tex
pdflatex main.tex
```

If using Overleaf, upload `main.tex` and the figure files from:

```bash
/home/wwww/projects/COSA-final-project/results/final_figures/
```

Then update figure paths in `main.tex` from `../../results/final_figures/...` to the uploaded figure filenames if needed.
