import pandas as pd
import QuantLib as ql
import unittest

def calculate_sensitivities(trades, 
                            bump_size=0.01, 
                            risk_free_rate=0.02, 
                            discount_factors=None, 
                            spot_prices=None, 
                            volatility_surfaces=None, 
                            fx_rates=None, 
                            bucket_mapping=None,
                            use_quantlib=False,
                            logger=None):
    """
    Calculate SIMM CRIF sensitivities for Equity Total Return Swaps (TRS).
    
    :param trades: List of trade strings in Quick Trade Format.
    :param bump_size: Sensitivity bump size (default 1% if not specified by SIMM CRIF).
    :param risk_free_rate: Risk-free rate used in calculations.
    :param discount_factors: DataFrame with columns ["Tenor", "DiscountFactor"] for discounting.
    :param spot_prices: Dict mapping equity/index to spot price.
    :param volatility_surfaces: Dict mapping equity/index to implied volatilities.
    :param fx_rates: Dict mapping currency to USD FX conversion rate.
    :param bucket_mapping: Dict mapping equity/index to SIMM bucket.
    :param use_quantlib: Flag to use QuantLib for pricing (default False).
    :param logger: Logger object for logging messages.
    :return: List of dictionaries in ISDA CRIF format.
    """
    # Initialize results
    sensitivities = []
    
    # Construct QuantLib discounting curve if needed
    if use_quantlib and discount_factors is not None:
        dates = [ql.Date().todaysDate() + ql.Period(int(tenor[:-1]), ql.Years) for tenor in discount_factors["Tenor"]]
        discounts = discount_factors["DiscountFactor"].values.tolist()
        curve = ql.DiscountCurve(dates, discounts, ql.Actual360())
        curve_handle = ql.YieldTermStructureHandle(curve)
    
    # Parse trades
    for trade in trades:
        try:
            parts = trade.split()
            quantity = float(parts[0])
            maturity_str = parts[1]
            equity_index = parts[2]
            trs_label = parts[3]
            direction = parts[4].lower()
            rate = float(parts[5].replace('bp', '')) / 10000 if 'bp' in parts[5] else float(parts[5])
            
            if trs_label.lower() != 'trs':
                raise ValueError("Invalid TRS trade format.")
            
            # Determine SIMM Bucket
            bucket = bucket_mapping.get(equity_index, 'Residual') if bucket_mapping else 'Residual'
            
            # Get spot price
            spot_price = spot_prices.get(equity_index, None)
            if spot_price is None:
                if logger:
                    logger.warning(f"Missing spot price for {equity_index}, using default value.")
                spot_price = 100.0  # Default placeholder value
            
            # Get FX conversion rate
            fx_rate = fx_rates.get("USD", 1.0)  # Default to USD if not provided
            if fx_rates and equity_index in fx_rates:
                fx_rate = fx_rates[equity_index]
            else:
                if logger:
                    logger.warning(f"Missing FX rate for {equity_index}, assuming USD.")
            
            # Convert maturity to numerical value in years
            if maturity_str.endswith('y'):
                maturity = int(maturity_str[:-1])
            else:
                raise ValueError("Unsupported maturity format.")
            
            # Determine position sign based on direction
            position_sign = 1 if direction == 'receive' else -1
            
            # Compute Delta Sensitivity (Risk_Equity)
            if use_quantlib:
                spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
                risk_free_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual360()))
                dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual360()))
                vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(volatility_surfaces.get(equity_index, 0.2))), ql.Actual360()))
                bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, risk_free_handle, vol_handle)
                
                # Define option parameters for delta calculation
                exercise = ql.EuropeanExercise(ql.Date().todaysDate() + ql.Period(maturity, ql.Years))
                payoff = ql.PlainVanillaPayoff(ql.Option.Call, spot_price)  # ATM option
                option = ql.VanillaOption(payoff, exercise)
                option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))
                
                # Calculate delta
                delta = option.delta() * spot_price * quantity * position_sign * fx_rate
            else:
                # Simple bump-and-revalue approach for delta
                bumped_up_price = spot_price * (1 + bump_size)
                bumped_down_price = spot_price * (1 - bump_size)
                delta = ((bumped_up_price - bumped_down_price) / (2 * spot_price)) * quantity * position_sign * fx_rate
            
            sensitivities.append({
                "RiskType": "Risk_Equity",
                "Qualifier": equity_index,
                "Bucket": bucket,
                "Label1": "",
                "Label2": "",
                "Amount": delta,
                "AmountCurrency": "USD",
                "AmountUSD": delta
            })
            
            # Compute Interest Rate Sensitivity (Risk_IRCurve)
            if use_quantlib and discount_factors is not None:
                df_maturity = curve.discount(dates[discount_factors["Tenor"].tolist().index(maturity_str)])
            else:
                df_maturity = 1 / ((1 + risk_free_rate) ** maturity)
            
            ir_sensitivity = -quantity * rate * df_maturity * position_sign * fx_rate
            
            sensitivities.append({
                "RiskType": "Risk_IRCurve",
                "Qualifier": "USD",  # Assume financing rate in USD
                "Bucket": "1",  # Default bucket
                "Label1": maturity_str,
                "Label2": "",
                "Amount": ir_sensitivity,
                "AmountCurrency": "USD",
                "AmountUSD": ir_sensitivity
            })
        
        except Exception as e:
            if logger:
                logger.critical(f"Error processing trade {trade}: {e}")
            continue
    
    return sensitivities




class TestCalculateSensitivities(unittest.TestCase):

    def setUp(self):
        self.trades = ["10000 5y sp500 trs pay 10bp"]
        self.spot_prices = {"sp500": 4000}
        self.volatility_surfaces = {"sp500": 0.2}
        self.fx_rates = {"USD": 1.0}
        self.bucket_mapping = {"sp500": "1"}
        self.discount_factors = pd.DataFrame({
            "Tenor": ["0y", "5y"],  # Include '0y' for the reference date
            "DiscountFactor": [1.0, 0.95]  # Discount factor of 1.0 at t=0
        })

    def test_without_quantlib(self):
        result = calculate_sensitivities(
            self.trades,
            spot_prices=self.spot_prices,
            volatility_surfaces=self.volatility_surfaces,
            fx_rates=self.fx_rates,
            bucket_mapping=self.bucket_mapping,
            use_quantlib=False
        )
        self.assertEqual(len(result), 2, f"Expected 2 risk types, got {len(result)}")
        self.assertEqual(result[0]["RiskType"], "Risk_Equity")
        self.assertEqual(result[1]["RiskType"], "Risk_IRCurve")

    def test_with_quantlib(self):
        result = calculate_sensitivities(
            self.trades,
            spot_prices=self.spot_prices,
            volatility_surfaces=self.volatility_surfaces,
            fx_rates=self.fx_rates,
            bucket_mapping=self.bucket_mapping,
            discount_factors=self.discount_factors,
            use_quantlib=True
        )
        self.assertEqual(len(result), 2, f"Expected 2 risk types, got {len(result)}")
        self.assertEqual(result[0]["RiskType"], "Risk_Equity")
        self.assertEqual(result[1]["RiskType"], "Risk_IRCurve")

if __name__ == "__main__":
    unittest.main()
