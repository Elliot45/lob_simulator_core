# animate_live.py
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lob_simulator import LOBSimulator, LOBConfig
from order_flow_models import PoissonLimitMarketCancel, OrderEvent, HawkesOrderFlow


def snapshot_levels(lob: LOBSimulator, depth: int = 10):
    snap = lob.snapshot(depth=depth)
    bids = snap["bids"]  # [(px, vol), ...] best->worse
    asks = snap["asks"]
    # pad
    if len(bids) < depth:
        bids = bids + [(float("nan"), 0.0)] * (depth - len(bids))
    if len(asks) < depth:
        asks = asks + [(float("nan"), 0.0)] * (depth - len(asks))
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
    ap.add_argument("--seconds", type=float, default=5.0, help="durée de l'animation")
    ap.add_argument("--dt_ms", type=float, default=1.0, help="pas de simu en millisecondes")
    ap.add_argument("--tick", type=float, default=0.01)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interval_ms", type=int, default=30, help="intervalle d'animation")
    ap.add_argument("--limit_bid", type=float, default=30.0)
    ap.add_argument("--limit_ask", type=float, default=30.0)
    ap.add_argument("--market_bid", type=float, default=6.0)
    ap.add_argument("--market_ask", type=float, default=6.0)
    ap.add_argument("--cancel_bid", type=float, default=24.0)
    ap.add_argument("--cancel_ask", type=float, default=24.0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # --- carnet + seed minimal
    cfg = LOBConfig(tick_size=args.tick, depth_levels=args.depth)
    lob = LOBSimulator(cfg)
    mid0 = 100.00
    for i in range(5):
        lob.add_limit_order("bid", mid0 - args.tick * (i + 1), 400 + 80 * i)
        lob.add_limit_order("ask", mid0 + args.tick * (i + 1), 400 + 80 * i)

    # --- modèle d'ordre flow
    if args.model == "poisson":
        model = PoissonLimitMarketCancel(
            limit_bid_lambda=args.limit_bid,
            limit_ask_lambda=args.limit_ask,
            market_bid_lambda=args.market_bid,
            market_ask_lambda=args.market_ask,
            cancel_bid_lambda=args.cancel_bid,
            cancel_ask_lambda=args.cancel_ask,
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
                "limit_bid": 2.0, "limit_ask": 2.0,
                "market_bid": 4.0, "market_ask": 4.0,
                "cancel_bid": 1.5, "cancel_ask": 1.5,
            },
            beta={
                "limit_bid": 5.0, "limit_ask": 5.0,
                "market_bid": 8.0, "market_ask": 8.0,
                "cancel_bid": 6.0, "cancel_ask": 6.0,
            },
        )

    # --- paramètres temps
    T = float(args.seconds)
    dt = args.dt_ms / 1000.0
    n_steps = int(T / dt)
    t_sim = 0.0

    # --- compteurs
    counters = {"limit": 0, "market": 0, "cancel": 0, "executed_qty": 0.0, "removed_qty": 0.0}

    # --- figure enrichie (carnet + mid/spread/imbalance)
    D = args.depth
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[2.0, 1.0], hspace=0.35, wspace=0.25)
    ax_book = fig.add_subplot(gs[0, :])
    ax_mid  = fig.add_subplot(gs[1, 0])
    ax_sp   = fig.add_subplot(gs[1, 1])
    ax_imb  = fig.add_subplot(gs[1, 2])

    ax_book.set_title("Order Book (live) — Bids à gauche, Asks à droite")
    ax_book.set_xlabel("Volume")
    ax_book.set_ylabel("Prix (L1 en haut)")
    ax_book.axvline(0, linewidth=0.8)
    

    # listes pour tracer mid/spread/imbalance au fil de l’eau
    t_hist, mid_hist, spr_hist, imb1_hist = [], [], [], []

    # init snapshot
    bids, asks = snapshot_levels(lob, D)
    y = np.arange(D)
    bids_px0 = [px for px, _ in bids]
    bids_vol0 = [v for _, v in bids]
    asks_px0 = [px for px, _ in asks]
    asks_vol0 = [v for _, v in asks]

    # barres miroir (bid volume négatif à gauche)
    bid_bars = ax_book.barh(y, [-v for v in bids_vol0], align="center", label="Bids")
    ask_bars = ax_book.barh(y, asks_vol0, align="center", label="Asks")

    ax_book.legend(loc="upper right")
    
    # libellés Y (prix Lk bid | ask)
    def set_ylabels(bids_px, asks_px):
        labels = []
        for k in range(D):
            bp = bids_px[k]
            ap = asks_px[k]
            bp_str = f"{bp:.2f}" if isinstance(bp, (int, float)) and not np.isnan(bp) else ""
            ap_str = f"{ap:.2f}" if isinstance(ap, (int, float)) and not np.isnan(ap) else ""
            if bp_str and ap_str:
                labels.append(f"{bp_str} | {ap_str}")
            elif bp_str:
                labels.append(f"{bp_str} |")
            elif ap_str:
                labels.append(f"| {ap_str}")
            else:
                labels.append("")
        ax_book.set_yticks(y)
        ax_book.set_yticklabels(labels)

    set_ylabels(bids_px0, asks_px0)

    # info text
    bb0, ba0, sp0, mid0_now = lob.best_bid(), lob.best_ask(), lob.spread(), lob.midprice()
    info_txt = ax_book.text(0.01, 0.98,
                            f"t = {t_sim:.3f}s\nbest_bid={bb0:.2f} | best_ask={ba0:.2f} | spread={sp0:.2f}",
                            transform=ax_book.transAxes, va="top", ha="left")

    # axes du bas (courbes live + curseurs verticaux)
    ax_mid.set_title("Δ Midprice (par rapport à t0)")   # <— on affiche l'écart
    ax_mid.set_xlabel("t (s)")
    ax_mid.set_ylabel("Δ price")
    ax_mid.ticklabel_format(style="plain", useOffset=False, axis="y")
    mid_line, = ax_mid.plot([], [], lw=1.0)
    mid_cursor = ax_mid.axvline(0.0, linestyle="--", linewidth=1.0)

    ax_sp.set_title("Spread")
    ax_sp.set_xlabel("t (s)")
    ax_sp.set_ylabel("spread")
    sp_line, = ax_sp.plot([], [], lw=1.0)
    sp_cursor = ax_sp.axvline(0.0, linestyle="--", linewidth=1.0)

    ax_imb.set_title("Imbalance L1")
    ax_imb.set_xlabel("t (s)")
    ax_imb.set_ylabel("imbalance")
    ax_imb.set_ylim(-0.05, 1.05)
    imb_line, = ax_imb.plot([], [], lw=1.0)
    imb_cursor = ax_imb.axvline(0.0, linestyle="--", linewidth=1.0)

    mid0_ref = None  # sera fixé au premier mid non-NaN


    # limites de volumes (on recalculera un plafond doux si besoin)
    max_vol = max(1.0, max(bids_vol0 + asks_vol0))
    ax_book.set_xlim(-1.1 * max_vol, 1.1 * max_vol)

    # --- générateur de frames (simu + snapshot) ---
    def step_generator():
        nonlocal t_sim, max_vol, mid0_ref
        for _ in range(n_steps):
            # 1) generarion des events
            evs = model.sample_events(t_sim, dt, lob)
            # 2) application
            for ev in evs:
                apply_event(lob, ev, counters)
            # 3) snapshot
            bids, asks = snapshot_levels(lob, D)
            bb, ba, sp, mid = lob.best_bid(), lob.best_ask(), lob.spread(), lob.midprice()

            # Référence pour Δmid
            if mid0_ref is None and mid is not None and not np.isnan(mid):
                mid0_ref = mid

            # 4) métriques série temporelle
            t_hist.append(t_sim)
            mid_hist.append(mid if mid is not None else np.nan)
            spr_hist.append(sp if sp is not None else np.nan)
            b1 = bids[0][1] if not np.isnan(bids[0][0]) else 0.0
            a1 = asks[0][1] if not np.isnan(asks[0][0]) else 0.0
            imb1 = b1 / (b1 + a1) if (b1 + a1) > 0 else np.nan
            imb1_hist.append(imb1)

            # 5) adapter l’échelle volume si ça gonfle
            cur_max = max([v for _, v in bids] + [v for _, v in asks] + [1.0])
            if cur_max > max_vol:
                max_vol = cur_max
                ax_book.set_xlim(-1.1 * max_vol, 1.1 * max_vol)

            yield {
                "t": t_sim,
                "bids_px": [px for px, _ in bids],
                "bids_vol": [v for _, v in bids],
                "asks_px": [px for px, _ in asks],
                "asks_vol": [v for _, v in asks],
                "bb": bb, "ba": ba, "sp": sp, "mid": mid
            }

            # 6) avancer le temps
            t_sim += dt

    # --- fonction update pour l’animation ---
    def update(state):
        bids_px, bids_vol = state["bids_px"], state["bids_vol"]
        asks_px, asks_vol = state["asks_px"], state["asks_vol"]
        bb, ba, sp, tt = state["bb"], state["ba"], state["sp"], state["t"]

        # barres
        for rect, v in zip(bid_bars, bids_vol):
            rect.set_width(-v)
        for rect, v in zip(ask_bars, asks_vol):
            rect.set_width(v)

        # labels de prix
        set_ylabels(bids_px, asks_px)

        # infos texte
        bb_s = f"{bb:.2f}" if bb is not None else "NA"
        ba_s = f"{ba:.2f}" if ba is not None else "NA"
        sp_s = f"{sp:.2f}" if sp is not None else "NA"
        info_txt.set_text(f"t = {tt:.3f}s\nbest_bid={bb_s} | best_ask={ba_s} | spread={sp_s}")

        # courbes bas : set data (Δmid)
        if mid0_ref is not None:
            mid_delta = [m - mid0_ref if (m is not None and not np.isnan(m)) else np.nan for m in mid_hist]
        else:
            mid_delta = [np.nan] * len(mid_hist)

        mid_line.set_data(t_hist, mid_delta)
        sp_line.set_data(t_hist, spr_hist)
        imb_line.set_data(t_hist, imb1_hist)

        # curseurs verticaux
        mid_cursor.set_xdata([tt, tt])
        sp_cursor.set_xdata([tt, tt])
        imb_cursor.set_xdata([tt, tt])

        # xlim auto-progressif
        for ax in (ax_mid, ax_sp, ax_imb):
            ax.set_xlim(0.0, max(tt, 1e-6))

        # autoscale Y raisonnable
        # mid: on met une marge autour du max |Δmid|
        if len(mid_delta) > 1:
            finite_mid = np.array([x for x in mid_delta if x == x])  # exclut NaN
            if finite_mid.size:
                pad = 0.1 * max(1e-6, np.nanmax(np.abs(finite_mid)))
                lo, hi = finite_mid.min() - pad, finite_mid.max() + pad
                if lo == hi:  # si plat, garde un peu d'amplitude
                    lo, hi = lo - 1e-6, hi + 1e-6
                ax_mid.set_ylim(lo, hi)

        # spread: autoscale doux
        if len(spr_hist) > 1:
            finite_sp = np.array([x for x in spr_hist if x == x])
            if finite_sp.size:
                pad = 0.1 * max(1e-6, np.nanmax(finite_sp))
                lo, hi = max(0.0, finite_sp.min() - pad), finite_sp.max() + pad
                if lo == hi:
                    lo, hi = 0.0, hi + 1e-6
                ax_sp.set_ylim(lo, hi)

        # imbalance: on reste [0,1]
        ax_imb.set_ylim(-0.05, 1.05)


        return (*bid_bars, *ask_bars, info_txt, mid_line, sp_line, imb_line,
                mid_cursor, sp_cursor, imb_cursor)

    anim = FuncAnimation(fig, update, frames=step_generator(),
                         interval=args.interval_ms, blit=False)

    plt.tight_layout()
    plt.show()

    # résumé
    print("=== Résumé ===")
    print("Événements:", counters)


if __name__ == "__main__":
    main()
