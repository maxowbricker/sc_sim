import sys
from data.loader import get_adapter
from simulator.simulation import run_simulation  # Ensure this exists and is implemented

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py simulation [1|2|...]")
        return

    sim_id = sys.argv[2]

    if sim_id == "1":
        adapter = get_adapter("didi", "./data/raw/didi")
    elif sim_id == "2":
        adapter = get_adapter("checkin", "./data/raw/weeplaces")
    else:
        print(f"Unknown simulation ID: {sim_id}")
        return

    run_simulation(adapter)


if __name__ == "__main__":
    main()