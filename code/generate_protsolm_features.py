import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import mdtraj as md
import pandas as pd
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm


DEFAULT_DATASET_ROOT = "/home/heyong/Protein_ADMET_SaProt/protsolm_saprot10k"

SS8_ALPHABET = ["G", "H", "I", "B", "E", "T", "S", "P", "L"]
SS3_ALPHABET = ["H", "E", "C"]
MAX_ASA = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "Q": 214.0,
    "E": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0,
}
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ProtSolM-compatible features on the local SaProt10k PDB set without relying on Bio.PDB DSSP."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke testing.")
    parser.add_argument("--executor", choices=["thread", "process"], default="process")
    return parser.parse_args()


def load_sequence_map(dataset_root):
    seq_map = {}
    for split in ["train", "valid", "test"]:
        split_csv = Path(dataset_root) / f"{split}.csv"
        with open(split_csv, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                seq_map[row["name"]] = row["aa_seq"]
    return seq_map


def properties_from_sequence(sequence):
    clean_seq = "".join([aa for aa in sequence if aa in "ACDEFGHIKLMNPQRSTVWY"])
    if not clean_seq:
        clean_seq = "G"

    counts = {
        "C": clean_seq.count("C"),
        "D": clean_seq.count("D"),
        "E": clean_seq.count("E"),
        "R": clean_seq.count("R"),
        "H": clean_seq.count("H"),
        "N": clean_seq.count("N"),
        "G": clean_seq.count("G"),
        "P": clean_seq.count("P"),
        "S": clean_seq.count("S"),
    }
    analyzer = ProteinAnalysis(clean_seq)
    length = len(clean_seq)

    return {
        "L": length,
        "1-C": counts["C"] / length,
        "1-D": counts["D"] / length,
        "1-E": counts["E"] / length,
        "1-R": counts["R"] / length,
        "1-H": counts["H"] / length,
        "Turn-forming residues fraction": (counts["N"] + counts["G"] + counts["P"] + counts["S"]) / length,
        "GRAVY": analyzer.gravy(),
    }


def compute_rsa_from_structure(structure):
    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    model = structure[0]
    sr.compute(model, level="R")

    rsa = []
    residue_names = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            aa = AA3_TO_1.get(residue.resname)
            if aa is None:
                continue
            sasa = float(getattr(residue, "sasa", 0.0))
            max_asa = MAX_ASA[aa]
            rsa.append(min(1.0, max(0.0, sasa / max_asa)))
            residue_names.append(aa)
    return rsa, residue_names


def properties_from_structure(pdb_path):
    traj = md.load(str(pdb_path))
    ss8 = md.compute_dssp(traj, simplified=False)[0].tolist()
    ss3 = md.compute_dssp(traj, simplified=True)[0].tolist()
    sasa = md.shrake_rupley(traj, mode="residue")[0]
    hbonds_num = int(md.kabsch_sander(traj)[0].nnz)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    rsa, residue_names = compute_rsa_from_structure(structure)

    plddt_values = [atom.get_bfactor() for atom in structure.get_atoms()]
    plddt = float(sum(plddt_values) / len(plddt_values)) if plddt_values else 0.0

    total_residues = min(len(ss8), len(ss3), len(sasa), len(rsa))
    if total_residues == 0:
        raise ValueError("No residues with structural features were extracted")

    ss8 = [state if state.strip() else "L" for state in ss8[:total_residues]]
    ss3 = [state if state.strip() else "C" for state in ss3[:total_residues]]
    rsa = rsa[:total_residues]

    properties = {}
    for state in SS8_ALPHABET:
        properties[f"ss8-{state}"] = sum(x == state for x in ss8) / total_residues
    for state in SS3_ALPHABET:
        properties[f"ss3-{state}"] = sum(x == state for x in ss3) / total_residues

    properties["Hydrogen bonds"] = hbonds_num
    properties["Hydrogen bonds per 100 residues"] = hbonds_num * 100.0 / total_residues

    for threshold in range(5, 105, 5):
        cutoff = threshold / 100.0
        exposed = sum(value >= cutoff for value in rsa)
        properties[f"Exposed residues fraction by {threshold}%"] = exposed / total_residues

    properties["pLDDT"] = plddt
    return properties


def process_one(name, seq, pdb_dir):
    pdb_path = Path(pdb_dir) / f"{name}.pdb"
    seq_props = properties_from_sequence(seq)
    struct_props = properties_from_structure(pdb_path)

    merged = {}
    merged.update(seq_props)
    merged.update(struct_props)
    merged["protein name"] = f"{name}.pdb"
    return merged


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_file = Path(args.out_file) if args.out_file else dataset_root / "SaProt10k_feature.csv"
    pdb_dir = dataset_root / "pdb"

    seq_map = load_sequence_map(dataset_root)
    items = list(seq_map.items())
    if args.limit > 0:
        items = items[: args.limit]
        seq_map = dict(items)
    rows = []
    errors = []

    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, name, seq, pdb_dir): name for name, seq in seq_map.items()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            name = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append((name, str(exc)))

    if not rows:
        preview = "\n".join(f"{name}: {err}" for name, err in errors[:10])
        raise RuntimeError(f"Feature extraction produced no rows. First errors:\n{preview}")

    df = pd.DataFrame(rows).sort_values("protein name").reset_index(drop=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} feature rows to {out_file}")

    if errors:
        error_file = out_file.with_suffix(".errors.csv")
        pd.DataFrame(errors, columns=["name", "error"]).to_csv(error_file, index=False)
        print(f"Skipped {len(errors)} proteins; details saved to {error_file}")


if __name__ == "__main__":
    main()
