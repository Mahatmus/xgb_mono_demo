# California Housing Dataset

**Source**: https://www.kaggle.com/datasets/camnugent/california-housing-prices
**Instances**: 20,640 | **Features**: 9 | **Target**: median_house_value

## Variables

| Variable | Type | Description | Monotonic Constraint |
|----------|------|-------------|---------------------|
| **longitude** | Continuous | Block group longitude | None |
| **latitude** | Continuous | Block group latitude | None |
| **housing_median_age** | Continuous | Median age of houses in block (lower = newer) | None |
| **total_rooms** | Integer | Total number of rooms within a block | None |
| **total_bedrooms** | Integer | Total number of bedrooms within a block | None |
| **population** | Integer | Total number of people residing within a block | None |
| **households** | Integer | Total number of households (people groups in home units) in block | None |
| **median_income** | Continuous | Median income for households in block (tens of thousands USD) | +1 |
| **ocean_proximity** | Categorical | Location of the house w.r.t ocean/sea | None |
| **median_house_value** | Continuous | **[TARGET]** Median house value in block (USD) | - |

## Notes

**median_income** is the only feature with an enforced monotonic constraint. Higher median income generally correlates with higher purchasing power and housing demand. However, this is an economic relationship rather than a physical law—edge cases exist (e.g., gentrifying areas, income vs. wealth distinctions).

Houses with median_house_value > $500,000 are removed as these represent censored/capped values in the original data.
