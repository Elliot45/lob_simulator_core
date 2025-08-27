from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional
import bisect

Side = Literal["bid","ask"]

@dataclass(frozen=True)
class LOBConfig:
    tick_size: float = 0.01
    depth_levels: int = 10 # L1...L10 par défaut
    price_min: float | None = None # bornes optionnelles (sandbox); Ancienne notation price_min: Optional[float] = None ici PEP 604
    price_max: float | None = None


class OrderBookSide:

    """
    Représentation 'price-level' (pas FIFO par ordre) pour un côté (bid/ask).
    - Pour bids: prix triés décroissants
    - Pour asks: prix triés croissants
    Stockage: dict price -> volume
    Un index trié (list) permet de retrouver rapidement best/browse profondeur.
    """

    def __init__(self,side : Side):
        self.side: Side = side
        self.levels: Dict[float, float] = {}
        self._prices_sorted: List[float] = [] # tri croissant; on inverse en Bid


    # -------------- internals --------------
    def _key_index(self, price: float) -> int:
        # tri crroissant; on utilise bisect pour inssérer/chercher
        return bisect.bisect_left(self._prices_sorted, price)
    
    def _insert_price(self,price: float) -> None:
        if price in self.levels:
            return
        idx = self._key_index(price)
        self._prices_sorted.insert(idx, price)
        self.levels[price] = 0.0

    def _remove_price_if_empty(self, price: float) -> None:
        vol = self.levels.get(price,0.0)
        if vol <= 0.0:
            if price in self.levels:
                del self.levels[price]
            idx = bisect.bisect_left(self._prices_sorted, price)
            if idx < len(self._prices_sorted) and self._prices_sorted[idx] == price:
                self._prices_sorted.pop(idx)
    

    # -------------- public API --------------
    def add_volume(self, price: float, qty: float) -> None:
        if qty <= 0:
            return
        self._insert_price(price)
        self.levels[price] += qty

    def remove_volume(self, price: float, qty: float) -> float:

        """
        Enlève 'qty' au niveau 'price'. Retourne la quantité effectivement retirée.
        """

        if qty <= 0 or price not in self.levels:
            return 0.0
        take = min(qty, self.levels[price])
        self.levels[price] -= take
        self._remove_price_if_empty(price)
        return take
    
    def best_price(self) -> float | None:
        if not self._prices_sorted:
            return None
        if self.side == "bid":
            return self._prices_sorted[-1] # max
        if self.side == "ask":
            return self._prices_sorted[0] # min
        
    def iter_levels(self, depth: int | None) -> List[Tuple[float,float]]:

        """
        Retourne [(price,volume)] trié du meilleur vers le plus loin, limité à 'depth' si fourni.
        """

        if self.side == "bid":
            prices = reversed(self._prices_sorted)
        if self.side == "ask":
            prices = iter(self._prices_sorted)

        out = []
        for p in prices:
            out.append((p,self.levels[p]))
            if depth is not None and len(out) >= depth:
                break
        return out
    
    def total_volume(self) -> float:
        return sum(self.levels.values())


class LOBSimulator:

    """
    Carnet d`ordres agrégé au niveau prix.
    - add_limit_order(side, price, qty): ajoute du volume au niveau de prix
    - add_market_order(side, qty): exécute contre le côté opposé en 'sweeping' les niveaux
    - cancel_order(side, price, qty): enlève du volume à un niveau
    """

    def __init__(self, config: LOBConfig | None):
        self.config = config or LOBConfig()
        self.bids = OrderBookSide("bid")
        self.asks = OrderBookSide("ask")


    # -------------- helpers --------------
    def _round_to_tick(self, price: float) -> float:
        ts = self.config.tick_size
        return round(round(price / ts) * ts, 10)
    
    def _valid_price(self, price: float) -> bool:
        lo = self.config.price_min
        hi = self.config.price_max
        if lo is not None and price < lo:
            return False
        if hi is not None and price > hi:
            return False
        return True
    
    def best_bid(self) -> float | None:
        return self.bids.best_price()
    
    def best_ask(self) -> float | None:
        return self.asks.best_price()
    
    def midprice(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0
    
    def spread(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return max(0.0, ba - bb)
    

    # -------------- core API --------------
    def add_limit_order(self, side: Side, price: float, qty: float) -> None:
        if qty <= 0:
            return
        p = self._round_to_tick(price)
        if not self._valid_price(p):
            raise ValueError(f"Price {p} out of bounds.")
        if side == "bid":
            # Un bid ne doit pas être > best ask si on veut empêcher crossing; on autorise ici (crossing = execution immédiate)
            if self.best_ask() is not None and p >= self.best_ask():
                # transforme l’excédent en exécution contre l’ask
                remaining = qty
                remaining = self._market_match("buy", remaining)
                if remaining > 0:
                    # Si crossing n’a pas tout exécuté et qu’on est au-dessus du meilleur ask (peu probable),
                    # on place le résiduel au niveau p (post-trade mid peut avoir bougé)
                    self.bids.add_volume(p, remaining)
            else:
                self.bids.add_volume(p, qty)
        else:
            if self.best_bid() is not None and p <= self.best_bid():
                remaining = qty
                remaining = self._market_match("sell", remaining)
                if remaining > 0:
                    self.asks.add_volume(p, remaining)
            else:
                self.asks.add_volume(p, qty)

    def add_market_order(self, side: Side, qty: float) -> float:
        if qty <= 0:
            return 0.0
        direction = "buy" if side == "bid" else "sell"
        executed = qty - self._market_match(direction, qty)
        return executed
    
    def cancel_order(self, side: Side, price: float, qty: float) -> float:
        p = self._round_to_tick(price)
        book = self.bids if side == "bid" else self.asks
        return book.remove_volume(p, qty)
    

    # -------------- matching --------------
    def _market_match(self, direction: Literal["buy","sell"], qty: float) -> float:

        """
        Exécute qty contre le côté opposé, en 'balayant' les niveaux du meilleur vers le pire.
        Retourne la quantité restante non exécutée.
        NB: exécution price-level (pas FIFO intra-niveau).
        """

        if qty <= 0:
            return 0.0
        
        if direction == "buy":
            # On hit le meilleur ask vers le haut
            while qty > 0:
                best = self.asks.best_price()
                if best is None:
                    break
                available = self.asks.levels[best]
                take = min(qty, available)
                self.asks.remove_volume(best, take)
                qty -= take
                if take == 0:
                    break
        
        else:
            # direction == "Sell": on hit le meilleur bid vers le bas
            while qty > 0:
                best = self.bids.best_price()
                if best is None:
                    break
                available = self.bids.levels[best]
                take = min(qty, available)
                self.bids.remove_volume(best, take)
                qty -= take
                if take == 0:
                    break

        return qty
    
    # -------------- snapshots --------------
    def snapshot(self, depth: int | None) -> Dict[str, List[Tuple[float,float]]]:

        """
        Renvoie une vue partielle du carnet:
          {"bids":[(price,vol),...], "asks":[(price,vol),...]}
        trié du meilleur vers le pire, limité à 'depth'.
        """

        return {
            "bids": self.bids.iter_levels(depth=depth),
            "asks": self.asks.iter_levels(depth=depth),
        }

    def assert_invariants(self) -> None:

        """
        Vérifications de base pour l’étape 1.
        """
        
        # volumes non négatifs
        for d in (self.bids.levels, self.asks.levels):
            for p, v in d.items():
                assert v >= 0.0, f"Negative volume at {p}"
        
        # pas d'overlap bid > ask
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            assert bb <= ba, f"Invariant broken: best_bid {bb} > best_ask {ba}"
