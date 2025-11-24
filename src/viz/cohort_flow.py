# src/viz/cohort_flow.py

def print_cohort_flow(steps):
    """
    steps = [
        ("Total participants", 44000),
        ("Has fundus images", 37000),
        ("Has eGFR or CKD data", 32000),
        ("Has complete survival follow-up", 30500),
    ]
    """
    print("COHORT FLOW")
    print("===========")
    for label, n in steps:
        print(f"{label}: {n}")
