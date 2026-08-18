"""Derive ground-truth CSV rows mechanically from PRIMARY project artifacts.

Sources (in provenance order):
  A. legacy {DS}_attack_to_nids.pt  — per-attack node ids used by all published evaluations
  B. legacy ground-truth .pt        — flat node-id list shipped with the project
  C. addition_list constants        — ids hardcoded in try_different_threshold.py (history)

Every emitted row is resolved index -> (table, uuid, content) against the ORIGINAL
database. Base rows = the pristine .bak backups of the team CSVs (pre any edit of ours).
Ids present only in C (not in A/B-attributable attacks) are NOT written into attack CSVs;
they go to a *_needs_review.csv for human adjudication. An AUDIT log records everything.
"""
import os, torch, psycopg2, shutil

GT = os.path.expanduser("~/guard/Ground_Truth/darpa/darpa")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _local_config():
    lc = os.path.join(REPO, "data_preparation", "local_config.json")
    if os.path.exists(lc):
        import json
        return json.load(open(lc))
    return {}


def _db_password():
    # Open-source note: real credentials come from data_preparation/local_config.json
    # (gitignored) or the PROVFUSION_DB_PASSWORD environment variable.
    return (_local_config().get("database", {}).get("password")
            or os.environ.get("PROVFUSION_DB_PASSWORD", "YOUR_DB_PASSWORD"))


# Only needed for `verify` mode / resolving legacy maps from the original repo.
_V3_DIR = (_local_config().get("v3_graphmae_dir")
           or os.environ.get("V3_GRAPHMAE_DIR", "/path/to/v3_GraphMAE"))


SPECS = {
    "THEIA_E3": {
        "db": "theia_e3",
        "attack_map": os.path.join(REPO, "THEIA_E3_attack_to_nids.pt.legacy"),
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids.pt"),
        "addition_list": [215236, 215806, 215179, 215721, 215237, 215796, 215235, 215854, 215801, 350388, 1027134],
        "attacks": [  # (attack_idx, csv_relpath, base = pristine backup or None for new)
            (0, "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv", "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv.bak_20260710"),
            (1, "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv", "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv"),
            (2, "E3-THEIA/node_phishing_email.csv", None),
        ],
    },
    "CLEARSCOPE_E5": {
        "db": "clearscope_e5",
        "attack_map": os.path.join(_V3_DIR, "CLEARSCOPE_E5_attack_to_nids.pt"),
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids_clearscope_e5.pt"),
        "addition_list": [158937, 445211],
        "attacks": [
            (0, "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv", "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv"),
            (1, "E5-CLEARSCOPE/node_clearscope_e5_fennec_vagrant_0517.csv", None),
            (2, "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv", "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv"),
            (3, "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv", "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv"),
        ],
    },
    "CLEARSCOPE_E3": {
        "db": "clearscope_e3",
        "attack_map": os.path.join(REPO, "CLEARSCOPE_E3_attack_to_nids.pt.legacy"),
        "legacy_gt": os.path.expanduser("~/guard/Ground_Truth/ground_truth_nids_clearscope.pt"),
        "addition_list": [195543, 198077, 198786, 195535, 198074, 198784, 287651, 287734],
        "attacks": [
            (0, "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv.bak_20260710"),
        ],
    },
}

def db_lookup(db):
    con = psycopg2.connect(database=db, host="localhost", user="postgres", password=_db_password(), port=5432)
    cur = con.cursor(); out = {}
    for t, cols, fmt in [
        ("subject_node_table", "path,cmd", lambda r: "{'subject': '%s %s'}" % (r[0] if r[0] is not None else "None", r[1] if r[1] is not None else "")),
        ("file_node_table", "path", lambda r: "{'file': '%s'}" % (r[0],)),
        ("netflow_node_table", "src_addr,src_port,dst_addr,dst_port", lambda r: "{'netflow': '%s:%s->%s:%s'}" % r),
    ]:
        cur.execute("SELECT index_id, node_uuid, %s FROM %s" % (cols, t))
        for row in cur.fetchall():
            out[int(row[0])] = (t, row[1], fmt(tuple(row[2:])))
    con.close(); return out

for ds, spec in SPECS.items():
    print("=" * 78); print("### DATASET", ds)
    att = torch.load(spec["attack_map"])
    legacy_gt = set(int(x) for x in torch.load(spec["legacy_gt"]))
    idx = db_lookup(spec["db"])
    audit = []
    attack_union = set()
    for ai, rel, base in spec["attacks"]:
        target = os.path.join(GT, rel)
        base_rows = []
        if base:
            with open(os.path.join(GT, base)) as f:
                base_rows = [l.rstrip("\n") for l in f if l.strip()]
        base_uuids = {l.split(",")[0] for l in base_rows}
        nids = [int(n) for n in att[ai]["nids"]] if ai in att else []
        attack_union.update(nids)
        derived = []
        for n in sorted(set(nids)):
            if n not in idx:
                audit.append("MISSING-IN-DB attack%d nid=%d" % (ai, n)); continue
            t, uuid, desc = idx[n]
            if uuid in base_uuids:
                continue  # already present in pristine team CSV
            derived.append("%s,%s,%d" % (uuid, desc, n))
            audit.append("DERIVED attack%d nid=%d uuid=%s src=attack_to_nids table=%s" % (ai, n, uuid, t))
        with open(target, "w") as f:
            for l in base_rows: f.write(l + "\n")
            for l in derived: f.write(l + "\n")
        print("  attack%d %s: base=%d derived=%d total=%d" % (ai, os.path.basename(rel), len(base_rows), len(derived), len(base_rows) + len(derived)))
    # B-source check: legacy GT ids outside every attack
    gt_orphans = legacy_gt - attack_union
    # C-source check: addition_list ids outside every attack
    add_orphans = [n for n in spec["addition_list"] if n not in attack_union]
    review = os.path.join(GT, spec["attacks"][0][1].split("/")[0], ds + "_needs_review.csv")
    with open(review, "w") as f:
        for n in sorted(gt_orphans):
            t, uuid, desc = idx.get(n, ("?", "?", "?"))
            f.write("%s,%s,%d,source=legacy_gt_only\n" % (uuid, desc, n))
            audit.append("REVIEW nid=%d uuid=%s src=legacy_gt_only (in GT .pt, in no attack)" % (n, uuid))
        for n in add_orphans:
            t, uuid, desc = idx.get(n, ("?", "?", "?"))
            f.write("%s,%s,%d,source=code_addition_list_only\n" % (uuid, desc, n))
            audit.append("REVIEW nid=%d uuid=%s src=code_addition_list_only (hardcoded in eval code, in no attack)" % (n, uuid))
    print("  needs_review: gt_orphans=%d addition_orphans=%d -> %s" % (len(gt_orphans), len(add_orphans), os.path.basename(review)))
    with open(os.path.join(GT, ds + "_AUDIT.log"), "w") as f:
        f.write("\n".join(audit) + "\n")
    print("  audit lines:", len(audit))
print("=" * 78); print("DERIVE_DONE")
