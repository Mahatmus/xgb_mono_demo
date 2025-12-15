# Medical Insurance Dataset

**Source**: https://www.kaggle.com/datasets/mirichoi0218/insurance
**Instances**: 1,338 | **Features**: 6 | **Target**: charges (Individual medical costs)

## Variables

| Variable | Type | Description | Monotonic Constraint |
|----------|------|-------------|---------------------|
| **age** | Integer | Age of primary beneficiary | +1 |
| **sex** | Categorical | Gender (female, male) | None |
| **bmi** | Continuous | Body Mass Index (kg/m²) | None |
| **children** | Integer | Number of dependents covered | None |
| **smoker** | Categorical | Smoking status (yes, no) | None |
| **region** | Categorical | US region (northeast, southeast, southwest, northwest) | None |
| **charges** | Continuous | **[TARGET]** Individual medical costs billed | - |

## Notes

**age** is the only feature with an enforced monotonic constraint. Healthcare utilization increases with age due to biological factors (higher rates of chronic conditions, more frequent screenings, increased medication needs) and actuarial fundamentals. This is a well-established relationship in insurance pricing.

**bmi** is explicitly *not* constrained despite intuition—the relationship between BMI and health costs is U-shaped or J-shaped (both underweight and obese individuals have elevated risks), not monotonic.
