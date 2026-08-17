"""Compare a language folder's labels.json against the team's canonical map.

Run this BEFORE uploading. If a slug maps to a different sign than the rest
of the team uses, your recordings merge into the wrong class and nothing
looks wrong afterwards -- the model just quietly learns the wrong thing.

USAGE
-----
    python scripts/check_labels.py data/sequences_v2/<your folder>/labels.json
"""
import json, sys
from pathlib import Path
CANON = {"sl_001": "ជម្រាប់សួរ", "sl_002": "អរគុណ", "sl_003": "ខុស",
         "sl_004": "ត្រូវ", "sl_005": "គ្រួសារ", "sl_006": "ប៉ា",
         "sl_007": "ម៉ាក់"}
p = Path(sys.argv[1])
mine = json.loads(p.read_text(encoding="utf-8"))
bad = False
for slug in sorted(set(CANON) | set(mine)):
    a, b = CANON.get(slug), mine.get(slug)
    mark = "OK " if a == b else "!! "
    if a != b:
        bad = True
    print(f"  {mark}{slug}   team: {a}   yours: {b}")
print("\nMISMATCH — fix before uploading." if bad else "\nAll labels match. Safe to upload.")
