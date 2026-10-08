#!/usr/bin/env bash
# Rebuild the LaTeX documents: Paper 1 (report/papers/race_of_clocks), the clinical paper (report/papers/clinical_events)
# and the evidence brief (investment/brief).
# Builds in a scratch copy under experiments/.texbuild (git-ignored) and copies the PDFs back.
# Optional TEXMFHOME: a local TeX tree for fonts missing from the system TeX (e.g. Lato and FontAwesome 5 when
# texlive-fonts-extra is not installed); the Dockerfile installs everything needed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
B=experiments/.texbuild; rm -rf "$B"; mkdir -p "$B/papers"
cp -r report/papers/race_of_clocks report/papers/clinical_events report/papers/figures "$B/papers/"; cp -r investment/brief "$B/brief"
if [ -d experiments/.texmf_local ]; then
  export TEXMFHOME="$ROOT/experiments/.texmf_local" TEXMFVAR="$ROOT/experiments/.texmf_local/var" TEXMFCONFIG="$ROOT/experiments/.texmf_local/config"
fi
(cd "$B/papers/race_of_clocks" && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex > build.log 2>&1) || { tail -20 "$B/papers/race_of_clocks/build.log"; exit 1; }
(cd "$B/papers/clinical_events" && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex > build.log 2>&1) || { tail -20 "$B/papers/clinical_events/build.log"; exit 1; }
(cd "$B/brief" && latexmk -pdf -interaction=nonstopmode -halt-on-error evidence_brief.tex > build.log 2>&1) || { tail -20 "$B/brief/build.log"; exit 1; }
cp "$B/papers/race_of_clocks/paper.pdf" report/papers/race_of_clocks/paper.pdf
cp "$B/papers/clinical_events/paper.pdf" report/papers/clinical_events/paper.pdf
cp "$B/brief/evidence_brief.pdf" investment/sleeping_machines_evidence_brief.pdf
echo "built: report/papers/race_of_clocks/paper.pdf investment/sleeping_machines_evidence_brief.pdf"
