from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="lob_trace.csv")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--interval_ms", type=int, default=30)
    ap.add_argument("--save", type=str, default="")  # ex: out.gif ou out.mp4
    args = ap.parse_args()

    D = int(args.depth)
    df = pd.read_csv(args.csv)

    # --- Vérification colonnes ---
    need_cols = ["t", "best_bid", "best_ask", "spread", "mid"] \
        + [f"bid_px_{i}" for i in range(1, D+1)] + [f"bid_vol_{i}" for i in range(1, D+1)] \
        + [f"ask_px_{i}" for i in range(1, D+1)] + [f"ask_vol_{i}" for i in range(1, D+1)]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonnes manquantes dans {args.csv} : {missing}")

    # --- Données principales ---
    t = df["t"].to_numpy()
    mid = df["mid"].to_numpy()
    spr = df["spread"].to_numpy()

    # Imbalance L1 = bid_vol_1 / (bid_vol_1 + ask_vol_1)
    den = (df["bid_vol_1"] + df["ask_vol_1"]).replace(0, np.nan)
    imb1 = (df["bid_vol_1"] / den).to_numpy()

    # Limites x pour les barres (volumes)
    max_bid_vol = df[[f"bid_vol_{k}" for k in range(1, D+1)]].max().max()
    max_ask_vol = df[[f"ask_vol_{k}" for k in range(1, D+1)]].max().max()
    max_vol = float(max(1.0, max_bid_vol, max_ask_vol))

    # --- Figure & layout (gridspec) ---
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[2.0, 1.0], hspace=0.35, wspace=0.25)

    ax_book = fig.add_subplot(gs[0, :])     # grand panneau carnet
    ax_mid  = fig.add_subplot(gs[1, 0])
    ax_sp   = fig.add_subplot(gs[1, 1])
    ax_imb  = fig.add_subplot(gs[1, 2])

    # --- Panneau carnet (barres horizontales miroir) ---
    ax_book.set_title("Order Book (L1–L10) — Bids à gauche, Asks à droite")
    ax_book.set_xlabel("Volume")
    ax_book.set_ylabel("Prix (L1 en haut)")

    # Prépare y positions pour D niveaux
    y_pos = np.arange(D)

    def frame_data(i: int):
        row = df.iloc[i]
        bids_px = [row[f"bid_px_{k}"] for k in range(1, D+1)]
        bids_vol = [row[f"bid_vol_{k}"] for k in range(1, D+1)]
        asks_px = [row[f"ask_px_{k}"] for k in range(1, D+1)]
        asks_vol = [row[f"ask_vol_{k}"] for k in range(1, D+1)]
        # Remplace NaN de prix par 0 volume pour l'affichage
        bids_vol = [0.0 if (px is None or (isinstance(px, float) and np.isnan(px))) else float(v) for px, v in zip(bids_px, bids_vol)]
        asks_vol = [0.0 if (px is None or (isinstance(px, float) and np.isnan(px))) else float(v) for px, v in zip(asks_px, asks_vol)]
        return bids_px, bids_vol, asks_px, asks_vol, float(row["best_bid"]), float(row["best_ask"]), float(row["spread"]), float(row["t"])

    # Frame initiale
    bids_px0, bids_vol0, asks_px0, asks_vol0, bb0, ba0, spr0, t0 = frame_data(0)

    # Barres (convention: volumes BID négatifs pour les dessiner à gauche)
    bid_bars = ax_book.barh(y_pos, [-v for v in bids_vol0], align="center", label="Bids")
    ask_bars = ax_book.barh(y_pos, asks_vol0, align="center", label="Asks")

    # Légendes et axe vertical central
    ax_book.legend(loc="upper right")
    ax_book.axvline(0, linewidth=0.8)

    # Libellés Y (prix Lk de chaque côté sur une même ligne)
    def set_ylabels(bids_px, asks_px):
        labels = []
        for k in range(D):
            bp = bids_px[k] if (bids_px[k] is not None and not (isinstance(bids_px[k], float) and np.isnan(bids_px[k]))) else ""
            ap = asks_px[k] if (asks_px[k] is not None and not (isinstance(asks_px[k], float) and np.isnan(asks_px[k]))) else ""
            if bp != "" and ap != "":
                labels.append(f"{bp:.2f} | {ap:.2f}")
            elif bp != "":
                labels.append(f"{bp:.2f} |")
            elif ap != "":
                labels.append(f"| {ap:.2f}")
            else:
                labels.append("")
        ax_book.set_yticks(y_pos)
        ax_book.set_yticklabels(labels)

    set_ylabels(bids_px0, asks_px0)

    # Limites de volume stables
    ax_book.set_xlim(-1.1 * max_vol, 1.1 * max_vol)

    # Texte info
    info_txt = ax_book.text(0.01, 0.98,
                            f"t = {t0:.3f}s\nbest_bid={bb0:.2f} | best_ask={ba0:.2f} | spread={spr0:.2f}",
                            transform=ax_book.transAxes, va="top", ha="left")

    # --- Sous-graphiques : courbes statiques + curseur vertical mobile ---
    # MIDPRICE
    ax_mid.plot(t, mid)
    ax_mid.set_title("Midprice")
    ax_mid.set_xlabel("t (s)")
    ax_mid.set_ylabel("price")
    # Désactive l'offset “+1e2” pour plus de lisibilité
    ax_mid.ticklabel_format(style="plain", useOffset=False, axis="y")
    mid_cursor = ax_mid.axvline(t0, linestyle="--", linewidth=1.0)

    # SPREAD
    ax_sp.plot(t, spr)
    ax_sp.set_title("Spread")
    ax_sp.set_xlabel("t (s)")
    ax_sp.set_ylabel("spread")
    sp_cursor = ax_sp.axvline(t0, linestyle="--", linewidth=1.0)

    # IMBALANCE L1
    ax_imb.plot(t, imb1)
    ax_imb.set_title("Imbalance L1 = bid_vol_1 / (bid_vol_1 + ask_vol_1)")
    ax_imb.set_xlabel("t (s)")
    ax_imb.set_ylabel("imbalance")
    ax_imb.set_ylim(-0.05, 1.05)
    imb_cursor = ax_imb.axvline(t0, linestyle="--", linewidth=1.0)

    def update(i: int):
        bids_px, bids_vol, asks_px, asks_vol, bb, ba, sp, tt = frame_data(i)

        # Met à jour les barres
        for rect, v in zip(bid_bars, bids_vol):
            rect.set_width(-v)
        for rect, v in zip(ask_bars, asks_vol):
            rect.set_width(v)

        # Met à jour les labels de prix
        set_ylabels(bids_px, asks_px)

        # Met à jour le texte
        info_txt.set_text(f"t = {tt:.3f}s\nbest_bid={bb:.2f} | best_ask={ba:.2f} | spread={sp:.2f}")

        # Déplace les curseurs verticaux
        mid_cursor.set_xdata([tt, tt])
        sp_cursor.set_xdata([tt, tt])
        imb_cursor.set_xdata([tt, tt])

        return (*bid_bars, *ask_bars, info_txt, mid_cursor, sp_cursor, imb_cursor)

    anim = FuncAnimation(fig, update, frames=len(df), interval=args.interval_ms, blit=False)

    if args.save:
        # GIF → pillow ; MP4 → ffmpeg
        if args.save.lower().endswith(".gif"):
            anim.save(args.save, writer="pillow", fps=max(1, int(1000/args.interval_ms)))
        else:
            anim.save(args.save, fps=max(1, int(1000/args.interval_ms)))
        print("Animation sauvegardée :", args.save)
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
