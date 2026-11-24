from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_rules_file_lists_rules_and_test_command():
    text = (ROOT / "RULES.md").read_text(encoding="utf-8")
    assert "Never push or publish" in text
    assert "Never expose secrets" in text
    assert "python -m pytest -q" in text


def test_secret_files_are_gitignored():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines


def test_dev_requirements_list_what_the_tests_import():
    lines = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    names = {l.split("<")[0].split(">")[0].split("=")[0].strip().lower() for l in lines if l and l[0] not in "#-"}
    assert {"numpy", "pillow", "pytest"} <= names
