import pandas as pd

df = pd.read_csv("lob_trace.csv")

# Invariants microstructure
assert (df["spread"] >= -1e-9).all(), "Spread négatif détecté"
assert (df["best_bid"] <= df["best_ask"]).all(), "best_bid > best_ask détecté"
mid_ok = (abs(df["mid"] - (df["best_bid"] + df["best_ask"]) / 2) < 1e-9).all()
assert mid_ok, "mid != (bid+ask)/2"

print("OK — invariants de base vérifiés sur", len(df), "pas de temps")
