"""Static consistency checks for the DeforTrack LaTeX manuscript."""

import argparse
import re
from collections import Counter
from pathlib import Path


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def comma_keys(text: str, command: str) -> set[str]:
    keys = set()
    for group in re.findall(rf"\\{command}\s*\{{([^}}]+)\}}", text):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def check_braces(text: str) -> None:
    depth = 0
    for index, char in enumerate(text):
        if char not in "{}" or (index and text[index - 1] == "\\"):
            continue
        depth += 1 if char == "{" else -1
        if depth < 0:
            raise ValueError(f"Unexpected closing brace at character {index}")
    if depth:
        raise ValueError(f"Unbalanced braces: final depth {depth}")


def check_environments(text: str) -> None:
    stack = []
    for match in re.finditer(r"\\(begin|end)\s*\{([^}]+)\}", text):
        operation, environment = match.groups()
        if operation == "begin":
            stack.append(environment)
        elif not stack or stack.pop() != environment:
            raise ValueError(f"Mismatched environment near {match.group(0)}")
    if stack:
        raise ValueError(f"Unclosed environments: {stack}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", required=True)
    parser.add_argument("--bib", required=True)
    args = parser.parse_args()

    tex = strip_comments(Path(args.tex).read_text(encoding="utf-8"))
    bib = Path(args.bib).read_text(encoding="utf-8")
    check_braces(tex)
    check_environments(tex)

    citations = comma_keys(tex, "cite")
    bibliography = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    missing_citations = sorted(citations - bibliography)

    labels = re.findall(r"\\label\s*\{([^}]+)\}", tex)
    references = comma_keys(tex, "ref")
    missing_labels = sorted(references - set(labels))
    duplicate_labels = sorted(key for key, count in Counter(labels).items() if count > 1)

    placeholders = []
    for pattern in (r"\[\?\]", r"(?<!\?)\?\?(?!\?)"):
        placeholders.extend(re.findall(pattern, tex))

    tex_dir = Path(args.tex).resolve().parent
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing_graphics = sorted(path for path in graphics if not (tex_dir / path).is_file())

    problems = []
    if missing_citations:
        problems.append(f"missing bibliography keys: {missing_citations}")
    if missing_labels:
        problems.append(f"missing labels: {missing_labels}")
    if duplicate_labels:
        problems.append(f"duplicate labels: {duplicate_labels}")
    if placeholders:
        problems.append(f"unresolved placeholders: {placeholders}")
    if missing_graphics:
        problems.append(f"missing graphics: {missing_graphics}")
    if problems:
        raise SystemExit("Static manuscript audit failed: " + "; ".join(problems))

    print(
        f"Static manuscript audit passed: {len(citations)} citations, "
        f"{len(bibliography)} bibliography entries, {len(labels)} labels, "
        f"{len(references)} references, and {len(graphics)} graphics."
    )


if __name__ == "__main__":
    main()
