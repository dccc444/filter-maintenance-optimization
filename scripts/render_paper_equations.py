from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "最终论文正文.md"
OUT = ROOT / "tmp" / "paper_equations"


def normalize(latex: str) -> tuple[str, str]:
    tag_match = re.search(r"\\tag\{([^}]+)\}", latex)
    tag = tag_match.group(1) if tag_match else ""
    latex = re.sub(r"\\tag\{[^}]+\}", "", latex)
    latex = latex.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
    latex = latex.replace(r"\begin{cases}", "")
    latex = latex.replace(r"\end{cases}", "")
    latex = latex.replace("&", "")
    latex = latex.replace(r"\\", r"\qquad")

    mapping = {
        "中维修": "M", "大维修": "L", "维护后": "post", "且": "and",
        "或": "or", "第": "k", "次": "", "年": "yr", "后": "post",
    }

    def repl_text(match):
        value = match.group(1)
        for src, dst in mapping.items():
            value = value.replace(src, dst)
        value = re.sub(r"[^A-Za-z0-9,+\- ]", "", value).strip() or "state"
        return rf"\mathrm{{{value}}}"

    latex = re.sub(r"\\text\{([^{}]*)\}", repl_text, latex)
    latex = re.sub(
        r"\\mathrm\{([^{}]*)\}",
        lambda m: repl_text(m) if re.search(r"[^\x00-\x7F]", m.group(1)) else m.group(0),
        latex,
    )
    latex = latex.replace(r"\operatorname", r"\mathrm")
    latex = latex.replace(r"\le ", r"\leq ").replace(r"\ge ", r"\geq ")
    latex = latex.replace(r"\le_", r"\leq_").replace(r"\ge_", r"\geq_")
    latex = latex.replace(r"\le.", r"\leq.").replace(r"\ge.", r"\geq.")
    latex = latex.replace(r"\mathbb E", r"\mathbb{E}")
    latex = re.sub(r"\\frac([0-9A-Za-z])\{([^{}]+)\}", r"\\frac{\1}{\2}", latex)
    latex = re.sub(r"\\mathcal\s+([A-Za-z])", r"\\mathcal{\1}", latex)
    latex = re.sub(r"\\(widehat|overline|widetilde)\s+([A-Za-z])", r"\\\1{\2}", latex)
    latex = latex.replace("{:}", ":")
    latex = re.sub(r"\s+", " ", latex).strip()
    return latex, tag


def equations():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    current = []
    inside = False
    for line in lines:
        if line.strip() == r"\[":
            inside = True
            current = []
        elif line.strip() == r"\]" and inside:
            yield "\n".join(current)
            inside = False
        elif inside:
            current.append(line)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for index, raw in enumerate(equations(), start=1):
        latex, tag = normalize(raw)
        formula = rf"${latex}$"
        fig = plt.figure(figsize=(9.0, 0.55), dpi=220)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
        ax.axis("off")
        try:
            ax.text(0.5, 0.52, formula, ha="center", va="center", fontsize=13, color="black")
            if tag:
                ax.text(0.985, 0.52, f"({tag})", ha="right", va="center", fontsize=11, color="black")
            fig.savefig(OUT / f"eq-{index:02d}.png", transparent=True, bbox_inches="tight", pad_inches=0.04)
        except Exception as exc:
            (OUT / f"eq-{index:02d}.txt").write_text(f"{raw}\n{exc}", encoding="utf-8")
        finally:
            plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
