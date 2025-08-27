from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import math
import random

Side = Literal["bid","ask"]
EventType = Literal["limit", "market", "cancel"]

@dataclass(frozen=True)
class OrderEvent:
    t: float
    etype: EventType    
    side: Side          # bid ask
    qty: float
    price: float | None        


# -------- utilitaires légers ----------
def sample_lognormal_qty(mean: float = 3.0, sigma: float = 0.75, min_q: float = 1.0, max_q: float = 5_000.0) -> float:
    q = float(np.random.lognormal(mean, sigma))
    return float(max(min_q, min(q, max_q)))

def geometric_distance(p: float = 0.35, max_k: int = 50) -> int:
    k = np.random.geometric(p)
    return int(min(max(1, k), max_k))

def choose_limit_price(side: Side, tick: float, best_bid: float | None, best_ask: float | None) -> float:
    # mid synthétique si besoin
    if best_bid is None and best_ask is None:
        mid = 100.0
    elif best_bid is None:
        mid = best_ask - 5 * tick
    elif best_ask is None:
        mid = best_bid + 5 * tick
    else:
        mid = 0.5 * (best_bid + best_ask)
    
    k = geometric_distance()
    if side == "bid":
        anchor = best_bid if best_bid is not None else (math.floor(mid / tick) * tick)
        price = anchor - k * tick
    else:
        anchor = best_ask if best_ask is not None else (math.ceil(mid / tick) * tick)
        price = anchor + k * tick
    
    return round(round(price / tick) * tick, 10)

def choose_cancel_price(side: Literal["bid","ask"], lob) -> float | None:

    """
    Choisit un niveau de prix EXISTANT côté 'side', pondéré par le volume.
    Retourne None s'il n'y a aucun niveau côté demandé.
    """

    book_side = lob.bids if side == "bid" else lob.asks
    if not book_side.levels:
        return None
    
    prices = np.array(list(book_side.levels.keys()), dtype=float)
    vols = np.array([max(1e-12, v) for v in book_side.levels.values()], dtype=float)
    probs = vols / vols.sum()
    return float(np.random.choice(prices, p=probs))

# -------- interface minimaliste ----------
class BaseOrderFlowModel:
    def sample_events(self, t: float, dt: float, lob) -> list[OrderEvent]:
        raise NotImplementedError

# -------- Poisson LIMIT MARKET ----------
class PoissonLimitMarketCancel(BaseOrderFlowModel):
    def __init__(self, 
                 limit_bid_lambda: float = 30.0, 
                 limit_ask_lambda: float = 30.0, 
                 market_bid_lambda: float = 6.0,    
                 market_ask_lambda: float = 6.0,    
                 cancel_bid_lambda: float = 24.0,
                 cancel_ask_lambda: float = 24.0,
                 tick_size: float = 0.01,):
        self.tick = tick_size
        self.lb = limit_bid_lambda      
        self.la = limit_ask_lambda
        self.mb = market_bid_lambda     
        self.ma = market_ask_lambda
        self.cb = cancel_bid_lambda
        self.ca = cancel_ask_lambda     
        
    def _poisson_n(self, rate: float, dt: float):
        return int(np.random.poisson(max(0.0, rate) * dt))
    
    def sample_events(self, t: float, dt: float, lob) -> list[OrderEvent]:
        events: list[OrderEvent] = []
        bb, ba = lob.best_bid(), lob.best_ask()

        # LIMITS
        # LIMIT BID
        for _ in range(self._poisson_n(self.lb, dt)):
            p = choose_limit_price("bid", self.tick, bb, ba)
            q = sample_lognormal_qty()
            events.append(OrderEvent(t = t, etype="limit", side="bid", qty=q, price=p))
        
        # LIMIT ASK
        for _ in range(self._poisson_n(self.la, dt)):
            p = choose_limit_price("ask", self.tick, bb, ba)
            q = sample_lognormal_qty()
            events.append(OrderEvent(t=t, etype="limit", side="ask", qty=q, price=p))


        # MARKETS
        # MARKET BID
        for _ in range(self._poisson_n(self.mb, dt)):
            q = sample_lognormal_qty()
            events.append(OrderEvent(t, "market", "bid", q, None))  # buy -> tape l'Ask

        # MARKET ASK
        for _ in range(self._poisson_n(self.ma, dt)):
            q = sample_lognormal_qty()
            events.append(OrderEvent(t, "market", "ask", q, None))  # sell -> tape le bid

        
        # CANCEL
        # CANCEL BID
        for _ in range(self._poisson_n(self.cb, dt)):
            p = choose_cancel_price("bid", lob)
            if p is not None:
                q = sample_lognormal_qty()
                events.append(OrderEvent(t, "cancel", "bid", q, p))

        # CANCEL ASK
        for _ in range(self._poisson_n(self.cb, dt)):
            p = choose_cancel_price("ask", lob)
            if p is not None:
                q = sample_lognormal_qty()
                events.append(OrderEvent(t, "cancel", "ask", q, p))

        return events
        