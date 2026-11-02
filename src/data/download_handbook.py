"""Download a subset of the GitLab Handbook into data/raw/<department>/."""

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_URL = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
CONTENT_PATH = "content/handbook"
DEPARTMENTS = ["engineering", "people-group", "finance", "legal", "security", "support"]
MAX_FILES_PER_DEPARTMENT = 60
MIN_FILE_BYTES = 1000  # skip near-empty index pages

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        repo = Path(tmp) / "handbook"
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(repo))
        paths = [f"{CONTENT_PATH}/{d}" for d in DEPARTMENTS]
        run("git", "sparse-checkout", "set", *paths, cwd=repo)

        for dept in DEPARTMENTS:
            source = repo / CONTENT_PATH / dept
            files = sorted(
                f for f in source.rglob("*.md") if f.stat().st_size >= MIN_FILE_BYTES
            )[:MAX_FILES_PER_DEPARTMENT]
            target = RAW_DIR / dept
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                # flatten the path into the filename to avoid name collisions
                name = "__".join(f.relative_to(source).parts)
                shutil.copy2(f, target / name)
            print(f"{dept}: {len(files)} files")


if __name__ == "__main__":
    main()
