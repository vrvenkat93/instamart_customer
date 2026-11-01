import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from instamart_copilot.config import DATA_RAW
from instamart_copilot.part_a import run_part_a
from instamart_copilot.part_b import run_part_b
from instamart_copilot.part_c import run_part_c
from instamart_copilot.part_d import run_part_d

def main():
    parser = argparse.ArgumentParser(description="Run Instamart Customer Intelligence & Support Copilot")
    parser.add_argument("--part", choices=["A","B","C","D"], help="Run one project part")
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW, help="Directory containing generated CSVs")
    parser.add_argument("--resume", action="store_true", help="Reserved for resuming an agent run by run_id via module API")
    args = parser.parse_args()
    parts = [args.part] if args.part else ["A","B","C","D"]
    if "A" in parts:
        run_part_a(args.data_dir)
    if "B" in parts:
        run_part_b()
    if "C" in parts:
        run_part_c(args.data_dir)
    if "D" in parts:
        run_part_d()
    print("Pipeline execution finished.")

if __name__ == "__main__":
    main()
