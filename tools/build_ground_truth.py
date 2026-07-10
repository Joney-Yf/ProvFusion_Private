"""Build index-based ground-truth artifacts from the canonical UUID CSVs.

For a given dataset + Postgres database, resolves every CSV node UUID to that
database's index_id and emits:
  - <out_prefix>_ground_truth_nids.pt   (sorted list of all attack-node indices)
  - <out_prefix>_attack_to_nids.pt      ({attack_i: {"nids": [...], "time_range": [...]}})

Attack index order == the CSV file order below (matches the legacy maps).
time_range is carried over from the legacy attack map of the same dataset.
"""
import sys, os, torch, psycopg2

GT_DIR = os.path.expanduser("~/guard/Ground_Truth/darpa/darpa")
DATASETS = {
    "THEIA_E3": {
        "csvs": ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                 "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                 "E3-THEIA/node_phishing_email.csv"],
        "legacy_map": "THEIA_E3_attack_to_nids.pt",
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids.pt"),
        "additions": [215236, 215806, 215719, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134],
    },
    "CLEARSCOPE_E3": {
        "csvs": ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv"],
        "legacy_map": "CLEARSCOPE_E3_attack_to_nids.pt",
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids_clearscope.pt"),
        "additions": [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734],
    },
}

def uuid_to_index(db):
    con = psycopg2.connect(database=db, host="localhost", user="postgres", password="yangfan", port=5432)
    cur = con.cursor(); m = {}
    for t in ["netflow_node_table", "subject_node_table", "file_node_table"]:
        cur.execute(f"SELECT node_uuid, index_id FROM {t};")
        for u, i in cur.fetchall():
            m[u] = int(i)
    con.close(); return m

def main(dataset, db, out_prefix, verify):
    spec = DATASETS[dataset]
    u2i = uuid_to_index(db)
    attack_map, all_nids, missing = {}, set(), []
    legacy = torch.load(spec["legacy_map"])
    for ai, rel in enumerate(spec["csvs"]):
        nids = []
        with open(os.path.join(GT_DIR, rel)) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                u = line.split(",")[0]
                if u in u2i:
                    nids.append(u2i[u])
                else:
                    missing.append((rel, u))
        tr = legacy[ai]["time_range"] if ai in legacy else None
        attack_map[ai] = {"nids": sorted(set(nids)), "time_range": tr}
        all_nids.update(nids)
        print(f"attack {ai} ({os.path.basename(rel)}): {len(set(nids))} nodes found")
    if missing:
        print(f"WARNING: {len(missing)} CSV uuids not found in db {db}:")
        for rel, u in missing[:10]: print("   ", rel, u)
    gt_list = sorted(all_nids)
    torch.save(gt_list, out_prefix + "_ground_truth_nids.pt")
    torch.save(attack_map, out_prefix + "_attack_to_nids.pt")
    print(f"saved {out_prefix}_ground_truth_nids.pt ({len(gt_list)} nodes) and _attack_to_nids.pt")

    if verify:
        old_gt = set(torch.load(spec["legacy_gt"]))
        expect = old_gt | set(spec["additions"])
        new_gt = set(gt_list)
        print("[VERIFY] new GT set == legacy GT ∪ addition_list:",
              "PASS" if new_gt == expect else
              f"DIFF (+{sorted(new_gt-expect)[:8]} / -{sorted(expect-new_gt)[:8]})")
        for ai in attack_map:
            old = set(legacy[ai]["nids"]) if ai in legacy else set()
            new = set(attack_map[ai]["nids"])
            if new == old:
                print(f"[VERIFY] attack {ai} nids identical ({len(new)})")
            else:
                print(f"[VERIFY] attack {ai}: +{sorted(new-old)} / -{sorted(old-new)} (old {len(old)} -> new {len(new)})")

if __name__ == "__main__":
    ds, db, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    verify = len(sys.argv) > 4 and sys.argv[4] == "verify"
    main(ds, db, outp, verify)
