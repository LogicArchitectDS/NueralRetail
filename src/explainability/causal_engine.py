import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

class CausalInferenceEngine:
    """
    A wrapper for DoWhy to perform causal analysis on retail data.
    Focuses on understanding the impact of interventions (e.g., price changes, 
    engagement frequency) on business outcomes (e.g., churn).
    """
    
    def estimate_frequency_on_churn(self, df: pd.DataFrame):
        """
        Estimates the causal effect of 'Frequency' on 'is_churned'.
        Confounder assumed: 'Monetary' (higher spend might correlate with frequency and churn risk).
        """
        # Ensure data is in correct format (DoWhy prefers bool/int for treatment/outcome)
        df = df.copy()
        
        # Define the Causal Graph
        model = CausalModel(
            data=df,
            treatment='Frequency',
            outcome='is_churned',
            common_causes=['Monetary'],
            effect_modifiers=[]
        )
        
        # 1. Identify causal effect and return target estimands
        identified_estimand = model.identify_effect(proceed_when_unidentified=True)
        
        # 2. Estimate the causal effect
        # We use a linear regression estimator for simplicity and stability in production
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        
        return {
            "causal_estimate": estimate.value,
            "treatment": "Frequency",
            "outcome": "is_churned",
            "confounders": ["Monetary"],
            "interpretation": f"On average, each additional purchase (Frequency) changes the churn probability by {estimate.value:.4f}."
        }

    def estimate_price_on_demand(self, historical_prices, historical_demands):
        """
        Estimates causal impact of price on demand using a simple linear model.
        """
        df = pd.DataFrame({
            'price': historical_prices,
            'demand': historical_demands
        })
        
        model = CausalModel(
            data=df,
            treatment='price',
            outcome='demand',
            common_causes=[] # Simplified for MVP
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentified=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        
        return {
            "causal_price_impact": estimate.value,
            "interpretation": f"Each $1 increase in price causes demand to change by {estimate.value:.2f} units."
        }
