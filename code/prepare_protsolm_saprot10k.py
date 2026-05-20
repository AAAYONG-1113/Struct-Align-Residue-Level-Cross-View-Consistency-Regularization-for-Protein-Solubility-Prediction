import argparse
import csv
import os
from pathlib import Path


DEFAULT_SOURCE_CSV = "/home/heyong/Protein_ADMET_SaProt/solubility_10000_SaProt_Ready.csv"
DEFAULT_PDB_3000_DIR = "/home/heyong/Protein_ADMET_SaProt/pdbs_3000"
DEFAULT_PDB_7000_DIR = "/home/heyong/Protein_ADMET_SaProt/pdbs_7000"
DEFAULT_OUTPUT_DIR = "/home/heyong/Protein_ADMET_SaProt/protsolm_saprot10k"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the local SaProt 10k solubility dataset into the CSV/PDB layout expected by ProtSolM."
    )
    parser.add_argument("--source-csv", default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--pdb-3000-dir", default=DEFAULT_PDB_3000_DIR)
    parser.add_argument("--pdb-7000-dir", default=DEFAULT_PDB_7000_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks by default to avoid duplicating 10k PDB files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_name_and_pdb(orig_idx, pdb_3000_dir, pdb_7000_dir):
    if orig_idx < 3000:
        name = f"seq_{orig_idx}"
        pdb_path = Path(pdb_3000_dir) / f"{name}.pdb"
    else:
        name = f"seq_7k_{orig_idx - 3000}"
        pdb_path = Path(pdb_7000_dir) / f"{name}.pdb"
    return name, pdb_path


def ensure_link(src, dst, overwrite=False):
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    os.symlink(src, dst)


def ensure_copy(src, dst, overwrite=False):
    if dst.exists() and not overwrite:
        return
    if dst.exists():
        dst.unlink()
    dst.write_bytes(src.read_bytes())


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    pdb_dir = output_dir / "pdb"
    esmfold_pdb_dir = output_dir / "esmfold_pdb"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    rows_by_stage = {"train": [], "valid": [], "test": []}
    missing_pdb = []

    with open(args.source_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for orig_idx, row in enumerate(reader):
            stage = row["stage"].strip().lower()
            if stage not in rows_by_stage:
                continue

            name, pdb_path = build_name_and_pdb(orig_idx, args.pdb_3000_dir, args.pdb_7000_dir)
            if not pdb_path.exists():
                missing_pdb.append(str(pdb_path))
                continue

            target_pdb = pdb_dir / f"{name}.pdb"
            if args.link_mode == "symlink":
                ensure_link(pdb_path, target_pdb, overwrite=args.overwrite)
            else:
                ensure_copy(pdb_path, target_pdb, overwrite=args.overwrite)

            rows_by_stage[stage].append(
                {
                    "name": name,
                    "aa_seq": row["protein"],
                    "label": int(float(row["label"])),
                    "orig_idx": orig_idx,
                }
            )

    if missing_pdb:
        preview = "\n".join(missing_pdb[:10])
        raise FileNotFoundError(f"Missing {len(missing_pdb)} PDB files. First entries:\n{preview}")

    for stage, rows in rows_by_stage.items():
        out_csv = output_dir / f"{stage}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "aa_seq", "label", "orig_idx"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {out_csv}")

    if esmfold_pdb_dir.exists() or esmfold_pdb_dir.is_symlink():
        if args.overwrite:
            esmfold_pdb_dir.unlink()
    if not esmfold_pdb_dir.exists():
        os.symlink(pdb_dir, esmfold_pdb_dir, target_is_directory=True)

    print(f"PDB directory ready at {pdb_dir}")


if __name__ == "__main__":
    main()
