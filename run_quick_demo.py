from lob_simulator import LOBSimulator, LOBConfig

if __name__ == "__main__":
    cfg = LOBConfig(tick_size=0.01, depth_levels=5)
    lob = LOBSimulator(cfg)

    # Seed book: 5 niveaux de chaque côté
    mid0 = 100.00
    for i in range(5):
        lob.add_limit_order("bid", mid0 - 0.01 * (i + 1), 50 + 10 * i)
        lob.add_limit_order("ask", mid0 + 0.01 * (i + 1), 50 + 10 * i)

    print("== SNAPSHOT L5 (seed) ==")
    print(lob.snapshot(depth=5))
    print("best_bid, best_ask, spread:", lob.best_bid(), lob.best_ask(), lob.spread())

    # Market buy 80 -> tape l’ask
    executed = lob.add_market_order("bid", 80)
    print("\nExecuted buy qty:", executed)
    print("== SNAPSHOT L5 (après market buy) ==")
    print(lob.snapshot(depth=5))
    print("best_bid, best_ask, spread:", lob.best_bid(), lob.best_ask(), lob.spread())

    # Limit sell crossing (place à 100.01 si best bid 99.99 → exécutée en partie/totalité)
    lob.add_limit_order("ask", 100.00, 120)  # crossing → devient une exécution
    print("\n== SNAPSHOT L5 (après limit sell crossing) ==")
    print(lob.snapshot(depth=5))
    print("best_bid, best_ask, spread:", lob.best_bid(), lob.best_ask(), lob.spread())

    # Cancel un niveau
    removed = lob.cancel_order("bid", lob.best_bid(), 9999)
    print("\nRemoved from best bid:", removed)
    print("== SNAPSHOT L5 (après cancel) ==")
    print(lob.snapshot(depth=5))
