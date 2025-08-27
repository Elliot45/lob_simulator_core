import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lob_trace.csv")

# Imbalance L1 : vol bid / (vol bid + vol ask)
imb1 = df["bid_vol_1"] / (df["bid_vol_1"] + df["ask_vol_1"]).replace(0, float("nan"))

# 1) Midprice
plt.figure()
plt.plot(df["t"], df["mid"])
plt.title("Midprice")
plt.xlabel("t (s)")
plt.ylabel("price")
plt.tight_layout()

# 2) Spread
plt.figure()
plt.plot(df["t"], df["spread"])
plt.title("Spread")
plt.xlabel("t (s)")
plt.ylabel("spread")
plt.tight_layout()

# 3) Imbalance L1
plt.figure()
plt.plot(df["t"], imb1)
plt.title("Order Book Imbalance (L1)")
plt.xlabel("t (s)")
plt.ylabel("imbalance")
plt.tight_layout()

plt.show()
