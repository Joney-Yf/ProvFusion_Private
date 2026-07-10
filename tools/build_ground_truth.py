"""Build index-based ground-truth artifacts from the canonical UUID CSVs.

For a given dataset + Postgres database, resolves every CSV node UUID to that
database's index_id and emits:
  - <out_prefix>_ground_truth_nids.pt   (sorted list of all attack-node indices)
  - <out_prefix>_attack_to_nids.pt      ({attack_i: {"nids": [...], "time_range": [...]}})

CSVs live in the repo's Ground_Truth_csv/ (node UUID is the canonical key; the trailing
index column is informational and valid only for the ORIGINAL databases). Attack index
order == the CSV order below and matches the legacy attack maps node-for-node (verified).
time_range values are ns-epoch attack windows copied verbatim from the legacy attack maps
(THEIA_E3 / CLEARSCOPE_E3 from this repo's pre-canonical maps at commit e5f40bb;
CLEARSCOPE_E5 from v3_GraphMAE/CLEARSCOPE_E5_attack_to_nids.pt) — recorded here so no
legacy .pt is needed at build time.

Usage:
    python tools/build_ground_truth.py <DATASET> <database> <out_prefix> [verify]
`verify` additionally compares against legacy label files where available (original DBs).
"""
import sys, os, torch, psycopg2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "Ground_Truth_csv")

DATASETS = {
    "THEIA_E3": {
        "csvs_and_windows": [
            ("E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",  [1523551200000000000, 1523554200000000000]),
            ("E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv", [1523385000000000000, 1523386800000000000]),
            ("E3-THEIA/node_phishing_email.csv",                    [1523641800000000000, 1523642700000000000]),
        ],
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids.pt"),
        "legacy_map": os.path.join(REPO, "THEIA_E3_attack_to_nids.pt"),
    },
    "CLEARSCOPE_E3": {
        "csvs_and_windows": [
            ("E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", [1523469240000000000, 1523472480000000000]),
        ],
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids_clearscope.pt"),
        "legacy_map": os.path.join(REPO, "CLEARSCOPE_E3_attack_to_nids.pt"),
    },
    "CLEARSCOPE_E5": {
        "csvs_and_windows": [
            ("E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",     [1557949080000000000, 1557951540000000000]),
            ("E5-CLEARSCOPE/node_clearscope_e5_fennec_vagrant_0517.csv", [1558108140000000000, 1558121520000000000]),
            ("E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",      [1558122480000000000, 1558123260000000000]),
            ("E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",         [1558124400000000000, 1558124880000000000]),
        ],
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids_clearscope_e5.pt"),
        "legacy_map": "/home/yangfan/guard/v3_GraphMAE/CLEARSCOPE_E5_attack_to_nids.pt",
    },
}


def uuid_to_index(db):
    con = psycopg2.connect(database=db, host="localhost", user="postgres", password="yangfan", port=5432)
    cur = con.cursor()
    m = {}
    for t in ["netflow_node_table", "subject_node_table", "file_node_table"]:
        cur.execute(f"SELECT node_uuid, index_id FROM {t};")
        for u, i in cur.fetchall():
            m[u] = int(i)
    con.close()
    return m


def build(dataset, db, out_prefix, verify=False):
    spec = DATASETS[dataset]
    u2i = uuid_to_index(db)
    attack_map, all_nids, missing = {}, set(), []
    for ai, (rel, window) in enumerate(spec["csvs_and_windows"]):
        nids = []
        with open(os.path.join(CSV_DIR, rel)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                u = line.split(",")[0]
                if u in u2i:
                    nids.append(u2i[u])
                else:
                    missing.append((rel, u))
        attack_map[ai] = {"nids": sorted(set(nids)), "time_range": window}
        all_nids.update(nids)
        print(f"attack {ai} ({os.path.basename(rel)}): {len(set(nids))} nodes")
    if missing:
        print(f"WARNING: {len(missing)} CSV uuids not found in db {db}:")
        for rel, u in missing[:10]:
            print("   ", rel, u)
    gt_list = sorted(all_nids)
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(gt_list, out_prefix + "_ground_truth_nids.pt")
    torch.save(attack_map, out_prefix + "_attack_to_nids.pt")
    print(f"saved {out_prefix}_ground_truth_nids.pt ({len(gt_list)} nodes) + _attack_to_nids.pt")

    if verify:
        try:
            old_gt = set(int(x) for x in torch.load(spec["legacy_gt"]))
            extra = old_gt - set(gt_list)
            print("[VERIFY] canonical ⊇ legacy GT:", "PASS" if not extra else f"FAIL missing {sorted(extra)[:8]}")
        except FileNotFoundError:
            print("[VERIFY] legacy GT not available — skipped")
        try:
            legacy = torch.load(spec["legacy_map"])
            for ai in attack_map:
                old = set(int(x) for x in legacy[ai]["nids"]) if ai in legacy else set()
                new = set(attack_map[ai]["nids"])
                tag = "identical (%d)" % len(new) if new == old else f"+{sorted(new-old)[:6]} / -{sorted(old-new)[:6]}"
                print(f"[VERIFY] attack {ai} vs legacy map: {tag}")
        except FileNotFoundError:
            print("[VERIFY] legacy map not available — skipped")
    return out_prefix


if __name__ == "__main__":
    ds, db, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    build(ds, db, outp, verify=(len(sys.argv) > 4 and sys.argv[4] == "verify"))
