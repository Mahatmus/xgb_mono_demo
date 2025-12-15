# Concrete Compressive Strength Dataset

**Source**: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
**Instances**: 1,030 | **Features**: 8 | **Target**: Concrete compressive strength (MPa)

## Variables

| Variable | Type | Unit | Description | Monotonic Constraint |
|----------|------|------|-------------|---------------------|
| **Cement** | Continuous | kg/m³ | Portland cement amount | None |
| **Blast Furnace Slag** | Integer | kg/m³ | Blast furnace slag amount | None |
| **Fly Ash** | Continuous | kg/m³ | Fly ash amount | None |
| **Water** | Continuous | kg/m³ | Water amount | None |
| **Superplasticizer** | Continuous | kg/m³ | Superplasticizer additive | None |
| **Coarse Aggregate** | Continuous | kg/m³ | Coarse aggregate amount | None |
| **Fine Aggregate** | Continuous | kg/m³ | Fine aggregate amount | None |
| **Age** | Integer | Days | Curing time (1-365 days) | +1 |
| **Compressive Strength** | Continuous | MPa | **[TARGET]** Concrete strength | - |

## Notes

**Age** is the only feature with an enforced monotonic constraint. Concrete gains strength through hydration—a chemical reaction between cement and water that continues over time. While the rate of strength gain slows significantly after 28 days, strength does not decrease with age under normal conditions. This is a physical/chemical guarantee, not merely an empirical correlation.
