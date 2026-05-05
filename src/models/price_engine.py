import numpy as np
import pandas as pd

class PriceIntelligenceEngine:
    """
    Analytical engine for Price Elasticity and Revenue Simulation (Amdox F-05).
    """

    def calculate_elasticity(self, historical_prices: list, historical_demands: list) -> dict:
        """
        Calculates the elasticity coefficient and R-squared using a log-log regression model.
        ln(Q) = alpha + beta * ln(P), where beta is the price elasticity.
        """
        # Ensure input arrays are numpy arrays and filter out non-positive values
        prices = np.array(historical_prices)
        demands = np.array(historical_demands)
        
        mask = (prices > 0) & (demands > 0)
        log_prices = np.log(prices[mask])
        log_demands = np.log(demands[mask])

        # Perform linear regression on log-transformed data
        # np.polyfit returns [slope, intercept]
        slope, intercept = np.polyfit(log_prices, log_demands, 1)
        
        # Calculate R-squared
        predicted_log_demands = slope * log_prices + intercept
        ss_res = np.sum((log_demands - predicted_log_demands) ** 2)
        ss_tot = np.sum((log_demands - np.mean(log_demands)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "elasticity_coefficient": float(slope),
            "r_squared": float(r_squared)
        }

    def simulate_revenue(self, current_price: float, current_demand: float, 
                         elasticity_coefficient: float, proposed_price: float) -> dict:
        """
        Simulates the demand and revenue impact of a proposed price change.
        """
        # Calculate percentage change in price
        pct_change_p = (proposed_price - current_price) / current_price
        
        # Calculate expected percentage change in demand based on elasticity
        # %ΔD = %ΔP * Elasticity
        pct_change_d = pct_change_p * elasticity_coefficient
        
        # Calculate new demand and projected revenue
        new_demand = current_demand * (1 + pct_change_d)
        projected_revenue = proposed_price * new_demand
        
        return {
            "new_demand": float(new_demand),
            "projected_revenue": float(projected_revenue)
        }

if __name__ == "__main__":
    # Instantiate the engine
    engine = PriceIntelligenceEngine()

    # Dummy historical data (Standard inverse relationship: price up, demand down)
    historical_prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    historical_demands = [1000, 850, 750, 650, 550, 500]

    # 1. Calculate Elasticity and R-squared
    metrics = engine.calculate_elasticity(historical_prices, historical_demands)
    elasticity = metrics["elasticity_coefficient"]
    r2 = metrics["r_squared"]

    # 2. Simulate Revenue Impact
    # Scenario: Current price is $12, demand is 750. We want to see impact of dropping to $11.
    current_price = 12.0
    current_demand = 750.0
    proposed_price = 11.0

    simulation = engine.simulate_revenue(
        current_price, 
        current_demand, 
        elasticity, 
        proposed_price
    )

    # Telemetry
    print("-" * 50)
    print("Amdox F-05 Price Intelligence Engine")
    print("-" * 50)
    print(f"Calculated Price Elasticity: {elasticity:.4f}")
    print(f"Model R-squared:             {r2:.4f}")
    print("-" * 25)
    print(f"Current State:  Price=${current_price}, Demand={current_demand}")
    print(f"Proposed State: Price=${proposed_price}")
    print("-" * 25)
    print(f"Simulated New Demand:    {simulation['new_demand']:.2f} units")
    print(f"Projected Total Revenue: ${simulation['projected_revenue']:.2f}")
    print("-" * 50)
