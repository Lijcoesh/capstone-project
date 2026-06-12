"""Quick diagnostic: print raw section content to see actual seizure label format."""
import re
from pathlib import Path

SUMMARY = Path("physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt")
EDFS = [
    "chb01_03.edf", "chb01_04.edf", "chb01_15.edf",
    "chb01_16.edf", "chb01_18.edf", "chb01_21.edf", "chb01_26.edf",
]

text = SUMMARY.read_text(encoding="utf-8", errors="ignore")

print("=== Raw section for chb01_03.edf ===")
match = re.search(r"File Name:\s*chb01_03\.edf\s*(.*?)(?=\nFile Name:|\Z)", text, re.S)
if match:
    print(repr(match.group(1)[:800]))
else:
    print("NO MATCH")

print()
print("=== Raw section for chb01_04.edf ===")
match = re.search(r"File Name:\s*chb01_04\.edf\s*(.*?)(?=\nFile Name:|\Z)", text, re.S)
if match:
    print(repr(match.group(1)[:800]))
else:
    print("NO MATCH")