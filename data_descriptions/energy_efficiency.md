# Energy Efficiency Dataset

**Source**: https://archive.ics.uci.edu/dataset/242/energy+efficiency
**Instances**: 768 | **Features**: 8 | **Targets**: Heating Load (Y1) and Cooling Load (Y2)

## Variables

| Variable | Type | Description | Monotonic Constraint | Exclude |
|----------|------|-------------|---------------------|---------|
| **Relative Compactness** | Continuous | Building compactness ratio | None | Yes |
| **Surface Area** | Continuous | Building surface area (m²) | +1 | No |
| **Wall Area** | Continuous | Wall surface area (m²) | None | Yes |
| **Roof Area** | Continuous | Roof surface area (m²) | None | No |
| **Overall Height** | Continuous | Building height (m) | None | No |
| **Orientation** | Integer | Building orientation (2-5) | None | No |
| **Glazing Area** | Continuous | Window area (m²) | None | No |
| **Glazing Area Distribution** | Integer | Window distribution (0-5) | None | No |
| **Heating Load** | Continuous | **[TARGET Y1]** Heating energy load | - | - |
| **Cooling Load** | Continuous | **[TARGET Y2]** Cooling energy load | - | - |

## Notes

**Surface Area** is the only feature with an enforced monotonic constraint (applies to both heating and cooling load targets). Heat transfer is proportional to surface area (Q = U·A·ΔT)—more surface means more heat loss in winter and more heat gain in summer. This is a thermodynamic guarantee.

## Excluded Variables

- **Relative Compactness**: Derived from surface area and volume; highly collinear with Surface Area
- **Wall Area**: Component of total surface area; redundant when Surface Area is included