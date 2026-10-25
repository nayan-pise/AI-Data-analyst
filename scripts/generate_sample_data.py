"""Generate the reproducible sample datasets: orders, customers, products.

``main(out_dir)`` writes three CSVs and returns the directory. The orders table carries
deliberately planted problems so the anomaly and data-quality features have something to find:

* 6 orders with absurd quantities (>= 40), 3 of them with 5-figure revenue
* 3 returns recorded as negative quantity / negative revenue
* 12 exact duplicate rows
* 15 missing discount values
* 1 order pointing at a customer id that does not exist in ``customers`` (orphaned key)
* two products (Kappa Cable, Mu Speaker) that sell far less than the rest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "sample_data"

PRODUCTS = [  # id, name, category, base price, unit cost, share of orders
    ("P001", "Alpha Laptop", "Computers", 1200, 840, 409),
    ("P002", "Beta Notebook", "Computers", 800, 560, 620),
    ("P003", "Gamma Tablet", "Computers", 450, 310, 360),
    ("P004", "Delta Monitor", "Displays", 300, 205, 405),
    ("P005", "Epsilon Headset", "Audio", 120, 70, 442),
    ("P006", "Zeta Keyboard", "Accessories", 80, 42, 452),
    ("P007", "Eta Mouse", "Accessories", 40, 18, 532),
    ("P008", "Theta Webcam", "Accessories", 90, 50, 281),
    ("P009", "Iota Dock", "Accessories", 150, 95, 288),
    ("P010", "Kappa Cable", "Accessories", 15, 5, 69),
    ("P011", "Lambda Lamp", "Office", 25, 11, 286),
    ("P012", "Mu Speaker", "Audio", 110, 72, 45),
]
REGIONS = {"East": 1102, "North": 892, "West": 890, "South": 786, "Central": 542}
SEGMENTS = {"Consumer": 0.45, "SMB": 0.35, "Enterprise": 0.20}
N_BASE, N_DUPES, N_CUSTOMERS = 4200, 12, 300


def _products() -> pd.DataFrame:
    return pd.DataFrame([{"product_id": i, "product_name": n, "category": c, "unit_price": p, "unit_cost": k}
                         for i, n, c, p, k, _ in PRODUCTS])


def _customers(rng: np.random.Generator) -> pd.DataFrame:
    ids = [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    return pd.DataFrame({
        "customer_id": ids,
        "customer_name": [f"Customer {i:03d}" for i in range(1, N_CUSTOMERS + 1)],
        "segment": rng.choice(list(SEGMENTS), N_CUSTOMERS, p=list(SEGMENTS.values())),
        "region": rng.choice(list(REGIONS), N_CUSTOMERS, p=np.array(list(REGIONS.values())) / sum(REGIONS.values())),
        "signup_date": (pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 730, N_CUSTOMERS), unit="D"))
        .strftime("%Y-%m-%d"),
    })


def _orders(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    n = N_BASE
    pw = np.array([p[5] for p in PRODUCTS], float)
    pid = rng.choice([p[0] for p in PRODUCTS], n, p=pw / pw.sum())
    base = {p[0]: p[3] for p in PRODUCTS}
    unit_price = np.round(np.array([base[x] for x in pid]) * rng.uniform(0.92, 1.09, n), 2)

    # dates 2024-01-01..2025-12-28 with growth and a Q4 / mid-year seasonal lift
    days = pd.date_range("2024-01-01", "2025-12-28")
    month = days.month.to_numpy()
    w = (1 + 0.012 * np.arange(len(days)) / 30) * np.where(month >= 10, 1.35, 1.0) * np.where(np.isin(month, [5, 6]), 1.15, 1.0)
    dates = rng.choice(days, n, p=w / w.sum())

    cust_w = rng.gamma(4.0, 1.0, N_CUSTOMERS)
    cid = rng.choice(customers["customer_id"], n, p=cust_w / cust_w.sum())
    qty = rng.choice([1, 2, 3, 4, 5], n, p=[0.5, 0.25, 0.13, 0.08, 0.04])
    disc = rng.choice([0.0, 0.05, 0.10, 0.15], n, p=[0.487, 0.161, 0.168, 0.184])
    rw = np.array(list(REGIONS.values()), float)
    region = rng.choice(list(REGIONS), n, p=rw / rw.sum())
    status = rng.choice(["Completed", "Shipped", "Cancelled"], n, p=[0.665, 0.162, 0.173])

    df = pd.DataFrame({
        "order_id": [f"O{100000 + i}" for i in rng.permutation(n) + 1],
        "order_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "customer_id": cid, "product_id": pid, "region": region,
        "quantity": qty, "unit_price": unit_price, "discount": disc, "status": status,
    }).sort_values(["order_date", "order_id"]).reset_index(drop=True)

    # --- planted anomalies (positions are random but reproducible) ---
    spots = rng.choice(n, 12, replace=False)
    big, neg, miss, orphan = spots[:6], spots[6:9], spots[9:10], spots[10]
    big_prod = ["P002", "P005", "P005", "P002", "P002", "P009"]
    for idx, p, q in zip(big, big_prod, [51, 63, 63, 65, 55, 45]):
        df.loc[idx, ["product_id", "quantity", "unit_price"]] = [p, q, round(base[p] * rng.uniform(0.95, 1.05), 2)]
        df.loc[idx, ["discount"]] = [0.0 if p == "P002" else 0.0]
    df.loc[neg, "quantity"] = -1
    df.loc[neg, "status"] = "Completed"
    df.loc[orphan, "customer_id"] = "C9999"

    df["revenue"] = np.round(df["quantity"] * df["unit_price"] * (1 - df["discount"]), 2)
    # the three high-value P002 bulk orders must stay 5-figure; the returns carry one big refund
    df.loc[neg[0], ["product_id", "unit_price", "revenue"]] = ["P001", 1187.5, -1187.5]
    df.loc[neg[0], "discount"] = 0.0

    # missing discounts (15) -- never on the planted rows
    safe = np.setdiff1d(np.arange(n), spots)
    df.loc[rng.choice(safe, 15, replace=False), "discount"] = np.nan

    dupes = df.loc[rng.choice(safe, N_DUPES, replace=False)]
    return pd.concat([df, dupes], ignore_index=True)


def main(out_dir: str | Path | None = None, seed: int = 42) -> Path:
    out = Path(out_dir) if out_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    customers = _customers(rng)
    orders = _orders(rng, customers)
    orders.to_csv(out / "orders.csv", index=False)
    customers.to_csv(out / "customers.csv", index=False)
    _products().to_csv(out / "products.csv", index=False)
    print("Wrote", ", ".join(sorted(p.name for p in out.glob("*.csv"))), "to", out)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
