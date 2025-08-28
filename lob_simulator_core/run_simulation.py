from __future__ import annotations
import numpy as np
import argparse
import pandas as pd

from lob_simulator import LOBConfig, LOBSimulator
from order_flow_models import PoissonLimitMarketCancel, OrderEvent, HawkesOrderFlow


def snapshot_levels(lob: LOBSimulator, depth: int = 10):
    snap = lob.snapshot(depth=depth)
    bids = snap["bids"]
    asks = snap["asks"]

    bids += [(np.nan, 0.0)] * max(0, depth - len(bids))
    asks += [(np.nan, 0.0)] * max(0, depth - len (asks))
    return bids[:depth], asks[:depth]


def apply_event(lob: LOBSimulator, ev: OrderEvent, counters: dict):

    if ev.etype == "limit":
        lob.add_limit_order(ev.side, float(ev.price), float(ev.qty))
        counters["limit"] += 1

    elif ev.etype == "market":
        executed = lob.add_market_order(ev.side, float(ev.qty))
        counters["market"] += 1
        counters["executed_qty"] += executed

    elif ev.etype == "cancel":
        removed = lob.cancel_order(ev.side, float(ev.price), float(ev.qty))
        counters["cancel"] += 1
        counters["removed_qty"] += removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["poisson","hawkes"], default="poisson")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--dt_ms", type=float, default=1.0)
    ap.add_argument("--tick", type=float, default=0.01)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="lob_trace.csv")
    ap.add_argument("--limit_bid",  type=float, default=30.0)
    ap.add_argument("--limit_ask",  type=float, default=30.0)
    ap.add_argument("--market_bid", type=float, default=6.0)
    ap.add_argument("--market_ask", type=float, default=6.0)
    ap.add_argument("--cancel_bid", type=float, default=24.0)
    ap.add_argument("--cancel_ask", type=float, default=24.0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    cfg = LOBConfig(tick_size=args.tick, depth_levels=args.depth)
    lob = LOBSimulator(cfg)

    mid0 = 100.0
    for i in range(5):
        lob.add_limit_order("bid", mid0 - args.tick * (i + 1), 400 + 80 * i)
        lob.add_limit_order("ask", mid0 + args.tick * (i + 1), 400 + 80 * i)

    if args.model == "poisson":
        model = PoissonLimitMarketCancel(
            limit_bid_lambda=30.0, limit_ask_lambda=30.0,
            market_bid_lambda=6.0, market_ask_lambda=6.0,
            cancel_bid_lambda=24.0, cancel_ask_lambda=24.0,
            tick_size=args.tick,
        )
    else:
        model = HawkesOrderFlow(
            tick_size=args.tick,
            mu={
                "limit_bid": args.limit_bid,
                "limit_ask": args.limit_ask,
                "market_bid": args.market_bid,
                "market_ask": args.market_ask,
                "cancel_bid": args.cancel_bid,
                "cancel_ask": args.cancel_ask,
            },
            alpha={
                "limit_bid": 2.0, "limit_ask": 2.0,   # excitation modérée pour les limit orders
                "market_bid": 4.0, "market_ask": 4.0, # excitation plus forte → rafales de market trades
                "cancel_bid": 1.5, "cancel_ask": 1.5, # excitation faible pour les cancels
            },
            beta={
                "limit_bid": 5.0, "limit_ask": 5.0,   # mémoire ~200 ms
                "market_bid": 8.0, "market_ask": 8.0, # mémoire courte ~125 ms
                "cancel_bid": 6.0, "cancel_ask": 6.0, # intermédiaire
            },
        )

    T = float(args.seconds)
    dt = args.dt_ms / 1000.0
    steps = int(T / dt)
    t = 0.0

    counters = {"limit": 0, "market": 0, "cancel": 0, "executed_qty": 0.0, "removed_qty": 0.0}
    rows = []

    for _ in range(steps):
        evs = model.sample_events(t, dt, lob)
        for ev in evs:
            apply_event(lob, ev, counters)
        bb, ba = lob.best_bid(), lob.best_ask()
        spr = lob.spread()
        mid = lob.midprice()
        bids, asks = snapshot_levels(lob, depth=args.depth)

        row = {
            "t": t,
            "best_bid": bb,
            "best_ask": ba,
            "spread": spr,
            "mid": mid,
            "cum_limit": counters["limit"],
            "cum_market": counters["market"],
            "cum_cancel": counters["cancel"],
            "cum_executed_qty": counters["executed_qty"],
            "cum_removed_qty": counters["removed_qty"],
        }

        for i, (px, vol) in enumerate(bids, start=1):
            row[f"bid_px_{i}"] = px
            row[f"bid_vol_{i}"] = vol
        for i, (px, vol) in enumerate(asks, start=1):
            row[f"ask_px_{i}"] = px
            row[f"ask_vol_{i}"] = vol

        rows.append(row)

        t += dt

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("=== Résumé ===")
    print("Événements:", counters)
    print("Trace sauvegardée:", args.out)
    print("Colonnes:", list(df.columns)[:12], "...")





if __name__ == "__main__":
    main()


