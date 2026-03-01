class TeaPredictionDTO:
    required_fields = [
        "Rainfall_mm",
        "Avg_Temperature_C",
        "Max_Temperature_C",
        "Min_Temperature_C",
        "Humidity_pct",
        "Sunshine_Hours",
        "Drought_Index",
        "USD_LKR",
        "Inflation_Rate",
        "Fuel_Price",
        "Interest_Rate",
        "Electricity_Cost",
        "Production_MT",
        "Auction_Quantity_MT",
        "Stocks_MT",
        "Plucking_Rate",
        "Fertilizer_Usage",
        "Labor_Cost",
        "Price_lag_1",
        "Price_lag_2",
        "Price_lag_3"
    ]

    @classmethod
    def validate(cls, data):
        missing = [f for f in cls.required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")
        return data