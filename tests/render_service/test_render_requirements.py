from pathlib import Path


def test_render_requirements_include_court_prediction_runtime_dependencies() -> None:
    requirements = Path("requirements-render.txt").read_text(encoding="utf-8").splitlines()

    assert "-r requirements-racketsport.txt" in requirements
    assert any(line.startswith("opencv-python-headless") for line in requirements)
