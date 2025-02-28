import pandas as pd
import numpy as np
import QuantLib as ql

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
            maturity = parts[1]
            equity_index = parts[2]
            trs_label = parts[3]
            direction = parts[4]
            rate = float(parts[5].replace('%', '')) / 100 if '%' in parts[5] else float(parts[5])
            
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
            
            # Compute Delta Sensitivity (Risk_Equity)
            if use_quantlib:
                spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
                risk_free_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual360()))
                dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual360()))
                vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(volatility_surfaces.get(equity_index, 0.2))), ql.Actual360()))
                bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, risk_free_handle, vol_handle)
                delta = bs_process.x0() * bump_size * quantity * fx_rate
            else:
                bumped_up = spot_price * (1 + bump_size)
                bumped_down = spot_price * (1 - bump_size)
                delta = ((bumped_up - bumped_down) / (2 * spot_price)) * quantity * fx_rate
            
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
        
        except Exception as e:
            if logger:
                logger.critical(f"Error processing trade {trade}: {e}")
            continue
    
    return sensitivities

# Unit Tests
def test_calculate_sensitivities():
    trades = ["10000 5y sp500 trs pay 10bp"]
    spot_prices = {"sp500": 4000}
    volatility_surfaces = {"sp500": 0.2}
    fx_rates = {"USD": 1.0}
    bucket_mapping = {"sp500": "1"}
    discount_factors = pd.DataFrame({"Tenor": ["5y"], "DiscountFactor": [0.95]})
    
    # Test without QuantLib
    result_no_ql = calculate_sensitivities(trades, spot_prices=spot_prices, 
                                           volatility_surfaces=volatility_surfaces, 
                                           fx_rates=fx_rates, bucket_mapping=bucket_mapping, 
                                           use_quantlib=False)
    assert len(result_no_ql) == 1, "Should return one risk type"
    
    # Test with QuantLib
    result_ql = calculate_sensitivities(trades, spot_prices=spot_prices, 
                                        volatility_surfaces=volatility_surfaces, 
                                        fx_rates=fx_rates, bucket_mapping=bucket_mapping, 
                                        discount_factors=discount_factors, 
                                        use_quantlib=True)
    assert len(result_ql) == 1, "Should return one risk type"
    
    print("All tests passed.")

test_calculate_sensitivities()
