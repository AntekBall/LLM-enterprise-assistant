from pathlib import Path


ROOT = Path(__file__).resolve().parent

DIRECTORIES = (
    ".github/workflows",
    "configs",
    "data/raw",
    "data/processed",
    "docker",
    "notebooks",
    "src/data",
    "src/retrieval",
    "src/models",
    "src/evaluation",
    "src/api",
    "tests",
)

FILES = (
    "configs/model_config.yaml",
    "configs/rag_config.yaml",
    "docker/Dockerfile",
    "docker/docker-compose.yml",
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_rag_experiment.ipynb",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/loader.py",
    "src/retrieval/__init__.py",
    "src/retrieval/vector_store.py",
    "src/models/__init__.py",
    "src/models/train_lora.py",
    "src/models/inference.py",
    "src/evaluation/__init__.py",
    "src/evaluation/evaluate_rag.py",
    "src/api/__init__.py",
    "src/api/main.py",
    "src/api/app.py",
    "tests/test_loader.py",
    "tests/test_retrieval.py",
    ".env.example",
    ".gitignore",
    "Makefile",
    "pyproject.toml",
    "README.md",
    ".github/workflows/ci-cd.yml",
)


def main() -> None:
    for directory in DIRECTORIES:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    for file_path in FILES:
        path = ROOT / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


if __name__ == "__main__":
    main()
