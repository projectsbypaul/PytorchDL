import pathlib
from typing import List

def _split_round_robin(items: List[pathlib.Path], n: int) -> List[List[pathlib.Path]]:
    buckets = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets

def _write_jobs(buckets: List[List[pathlib.Path]], outdir: pathlib.Path, width: int, mode: str, root: pathlib.Path, abs_paths: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    for idx, bucket in enumerate(buckets, start=1):
        jobname = outdir / f"Instance{idx:0{width}d}.job"
        with jobname.open("w", encoding="utf-8") as f:
            if mode == "dirs":
                for p in bucket:
                    f.write(p.name + "\n")
            else:
                for p in bucket:
                    f.write(str(p.resolve() if abs_paths else p.relative_to(root)) + "\n")

def make_jobs_dirs(root: str, instances: int, outdir: str = ".", abs_paths: bool = False):
    """Split immediate subdirectories into InstanceXXX.job files."""
    root = pathlib.Path(root).resolve()
    outdir = pathlib.Path(outdir).resolve()
    items = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    buckets = _split_round_robin(items, instances)
    width = max(3, len(str(instances)))
    _write_jobs(buckets, outdir, width, "dirs", root, abs_paths)

def make_jobs_ext(root: str, instances: int, extensions: List[str], outdir: str = ".", abs_paths: bool = False, recursive: bool = False):
    """Split files matching given extensions into InstanceXXX.job files."""
    root = pathlib.Path(root).resolve()
    outdir = pathlib.Path(outdir).resolve()
    exts = [(e if e.startswith(".") else "." + e).lower() for e in extensions]
    it = root.rglob("*") if recursive else root.glob("*")
    items = sorted([p for p in it if p.is_file() and p.suffix.lower() in exts], key=lambda p: str(p).lower())
    buckets = _split_round_robin(items, instances)
    width = max(3, len(str(instances)))
    _write_jobs(buckets, outdir, width, "ext", root, abs_paths)

def make_jobs_all(root: str, instances: int, outdir: str = ".", abs_paths: bool = False, recursive: bool = False):
    """Split all files into InstanceXXX.job files."""
    root = pathlib.Path(root).resolve()
    outdir = pathlib.Path(outdir).resolve()
    it = root.rglob("*") if recursive else root.glob("*")
    items = sorted([p for p in it if p.is_file()], key=lambda p: str(p).lower())
    buckets = _split_round_robin(items, instances)
    width = max(3, len(str(instances)))
    _write_jobs(buckets, outdir, width, "all", root, abs_paths)