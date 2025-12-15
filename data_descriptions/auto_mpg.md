# Auto MPG Dataset

**Source**: https://archive.ics.uci.edu/dataset/9/auto+mpg
**Instances**: 398 | **Features**: 7 | **Target**: mpg (City-cycle fuel consumption)

## Variables

| Variable | Type | Description | Monotonic Constraint |
|----------|------|-------------|---------------------|
| **cylinders** | Integer | Number of engine cylinders | None |
| **displacement** | Continuous | Engine displacement | None |
| **horsepower** | Continuous | Engine horsepower (has missing values) | None |
| **weight** | Continuous | Vehicle weight | -1 |
| **acceleration** | Continuous | Acceleration performance | None |
| **model_year** | Integer | Model year | None |
| **origin** | Integer | Origin (1=USA, 2=Europe, 3=Japan) | None |
| **manufacturer** | Categorical | Manufacturer name (extracted from car_name, 37 unique) | None |
| **mpg** | Continuous | **[TARGET]** Miles per gallon | - |

## Notes

**Weight** is the only feature with an enforced monotonic constraint. Heavier vehicles require more energy to accelerate and maintain motion, guaranteeing higher fuel consumption. This is a physical constraint (F=ma), not merely an empirical correlation.

**Manufacturer** is extracted from the first word of the original `car_name` field (e.g., "ford pinto" → "ford"). This reduces cardinality from 305 unique names to 37 manufacturers while preserving meaningful signal about design philosophy and fuel efficiency patterns.
