import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    if sys.argv[1:2] == ["compare"]:
        from vessel_power.model_comparison import main as compare

        compare(sys.argv[2:])
    elif sys.argv[1:2] == ["changes"]:
        from vessel_power.change_analysis import main as changes

        changes(sys.argv[2:])
    else:
        from vessel_power.pipeline import main as pipeline

        pipeline()


if __name__ == "__main__":
    main()
