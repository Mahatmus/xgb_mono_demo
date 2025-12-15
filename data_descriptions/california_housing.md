# California Housing Dataset

**Source**: sklearn.datasets.fetch_california_housing
**Instances**: 20,640 | **Features**: 8 | **Target**: MedHouseVal (Median house value in $100,000s)

## Variables

| Variable | Type | Description | Monotonic Constraint |
|----------|------|-------------|---------------------|
| **MedInc** | Continuous | Median income in block group | +1 |
| **HouseAge** | Continuous | Median house age in block group | None |
| **AveRooms** | Continuous | Average rooms per household | None |
| **AveBedrms** | Continuous | Average bedrooms per household | None |
| **Population** | Integer | Block group population | None |
| **AveOccup** | Continuous | Average household members | None |
| **Latitude** | Continuous | Block group latitude | None |
| **Longitude** | Continuous | Block group longitude | None |
| **MedHouseVal** | Continuous | **[TARGET]** Median house value | - |

## Notes

**MedInc** is the only feature with an enforced monotonic constraint. Higher median income generally correlates with higher purchasing power and housing demand. However, this is an economic relationship rather than a physical law—edge cases exist (e.g., gentrifying areas, income vs. wealth distinctions).