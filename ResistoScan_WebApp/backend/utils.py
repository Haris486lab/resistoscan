def risk_level(score):
    if score > 25000:
        return "CRITICAL"
    elif score > 20000:
        return "VERY HIGH"
    elif score > 15000:
        return "HIGH"
    else:
        return "MODERATE"
