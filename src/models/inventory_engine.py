import numpy as np
from scipy.stats import norm
import pandas as pd

class InventoryEngine:
    """
    Analytical math engine for Inventory Optimization (Amdox F-06).
    Provides methods for Economic Order Quantity, Safety Stock, and Reorder Point.
    """

    def calculate_eoq(self, annual_demand: float, order_cost: float, holding_cost: float) -> int:
        """
        Calculates the Economic Order Quantity (EOQ).
        Formula: sqrt((2 * D * S) / H)
        """
        if holding_cost <= 0:
            return 0
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
        return int(round(eoq))

    def calculate_safety_stock(self, max_lead_time: float, avg_lead_time: float,
                                max_daily_demand: float, avg_daily_demand: float) -> int:
        """
        Calculates Safety Stock using the Max-Avg formula.
        Formula: (Max Daily Demand * Max Lead Time) - (Avg Daily Demand * Avg Lead Time)
        """
        safety_stock = (max_daily_demand * max_lead_time) - (avg_daily_demand * avg_lead_time)
        return int(round(max(0, safety_stock)))

    def calculate_reorder_point(self, avg_daily_demand: float, avg_lead_time: float,
                                 safety_stock: int) -> float:
        """
        Calculates the Reorder Point (ROP).
        Formula: (Avg Daily Demand * Avg Lead Time) + Safety Stock
        """
        reorder_point = (avg_daily_demand * avg_lead_time) + safety_stock
        return float(reorder_point)

    def abc_xyz_classify(self, df: pd.DataFrame,
                         revenue_col: str = "revenue",
                         sku_col: str = "sku_id",
                         demand_col: str = "demand") -> pd.DataFrame:
        """
        ABC-XYZ Classification for SKU portfolio management.

        ABC: Revenue contribution tiers
            A = Top 80% of cumulative revenue (highest value SKUs)
            B = Next 15% of cumulative revenue
            C = Bottom 5% of cumulative revenue (long-tail SKUs)

        XYZ: Demand variability tiers
            X = Coefficient of Variation < 0.5 (stable, predictable demand)
            Y = CV between 0.5 and 1.0 (moderate variability)
            Z = CV > 1.0 (highly erratic, hard to forecast)

        Returns DataFrame with ABC class, XYZ class, and combined ABC-XYZ label.
        """
        result = df.copy()

        # ── ABC Classification ────────────────────────────────────────────
        result = result.sort_values(revenue_col, ascending=False)
        result["cumulative_revenue"] = result[revenue_col].cumsum()
        total_revenue = result[revenue_col].sum()
        result["cumulative_pct"] = result["cumulative_revenue"] / total_revenue * 100

        def abc_label(pct):
            if pct <= 80:
                return "A"
            elif pct <= 95:
                return "B"
            else:
                return "C"

        result["ABC"] = result["cumulative_pct"].apply(abc_label)

        # ── XYZ Classification ────────────────────────────────────────────
        if demand_col in result.columns:
            # If demand column contains lists/arrays of historical demand
            if result[demand_col].apply(lambda x: hasattr(x, '__len__')).all():
                result["demand_cv"] = result[demand_col].apply(
                    lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0
                )
            else:
                # Scalar demand — default to Y if no variance data
                result["demand_cv"] = 0.5
        else:
            result["demand_cv"] = 0.5

        def xyz_label(cv):
            if cv < 0.5:
                return "X"
            elif cv <= 1.0:
                return "Y"
            else:
                return "Z"

        result["XYZ"] = result["demand_cv"].apply(xyz_label)
        result["ABC_XYZ"] = result["ABC"] + result["XYZ"]

        # ── Summary ───────────────────────────────────────────────────────
        summary = result.groupby("ABC_XYZ").agg(
            sku_count=(sku_col, "count") if sku_col in result.columns else (result.index, "count"),
            total_revenue=(revenue_col, "sum")
        ).reset_index()
        summary["revenue_pct"] = (summary["total_revenue"] / total_revenue * 100).round(2)

        print("\n=== ABC-XYZ Classification Summary ===")
        print(summary.to_string(index=False))
        print(f"\nAX (High value, stable): Premium reorder candidates")
        print(f"AZ (High value, erratic): Safety stock priority")
        print(f"CZ (Low value, erratic): Liquidation candidates")

        return result[["ABC", "XYZ", "ABC_XYZ", "demand_cv", "cumulative_pct"]]

if __name__ == "__main__":
    # Instantiate the engine
    engine = InventoryEngine()

    # Dummy test values (Reasonable retail scenario)
    annual_demand = 12000.0
    order_cost = 50.0
    holding_cost = 2.5
    
    avg_daily_demand = annual_demand / 365
    max_daily_demand = avg_daily_demand * 1.5
    
    avg_lead_time = 7.0  # days
    max_lead_time = 12.0 # days

    # Run Calculations
    eoq = engine.calculate_eoq(annual_demand, order_cost, holding_cost)
    ss = engine.calculate_safety_stock(max_lead_time, avg_lead_time, max_daily_demand, avg_daily_demand)
    rop = engine.calculate_reorder_point(avg_daily_demand, avg_lead_time, ss)

    # Telemetry
    print("-" * 40)
    print("Amdox F-06 Inventory Optimization Engine")
    print("-" * 40)
    print(f"Annual Demand:  {annual_demand}")
    print(f"Order Cost:     ${order_cost}")
    print(f"Holding Cost:   ${holding_cost}")
    print("-" * 20)
    print(f"Calculated EOQ:          {eoq} units")
    print(f"Calculated Safety Stock: {ss} units")
    print(f"Calculated Reorder Point: {rop:.2f} units")
    print("-" * 40)
