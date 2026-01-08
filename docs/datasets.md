Regression datasets for XGBoost monotonic constraints
Finding the right datasets to demonstrate monotonic constraint techniques requires identifying regression problems where predictor-target relationships follow clear, intuitive directional patterns. Six datasets stand out across housing, insurance, automotive, gemology, and engineering domains—each offering strong candidates for monotonic constraints with documented relationships that any practitioner can understand.
California Housing offers the cleanest demonstration
The California Housing dataset from scikit-learn provides 20,640 samples with 8 features scikit-learn predicting median house values. The MedInc (median income) feature has a correlation of ~0.69 with house price—the strongest predictor and an obvious candidate for positive monotonic constraint. Higher neighborhood income should always predict higher prices.
Source: Built into scikit-learn (sklearn.datasets.fetch_california_housing)
Key monotonic variables:

MedInc (+1): Wealthier neighborhoods command premium prices
AveRooms (+1): Larger homes with more rooms are worth more
AveBedrms (+1): More bedrooms indicate higher value

This dataset appears in XGBoost's official monotonic constraints tutorials and requires no preprocessing—making it ideal for initial demonstrations.
Auto MPG showcases negative constraints
The UCI Auto MPG dataset (398 rows, 7 features) predicts fuel efficiency UCI Machine Learning Repository where physics dictates several negative monotonic relationships—heavier vehicles and larger engines consume more fuel.
Source: https://archive.ics.uci.edu/dataset/9/auto+mpg
Key monotonic variables:

weight (-1): Heavier vehicles require more energy, reducing MPG
displacement (-1): Larger engines burn more fuel
horsepower (-1): More power means more fuel consumption
cylinders (-1): Each additional cylinder adds fuel use
model_year (+1): Technological improvements increased efficiency through the 1970s-80s

This classic dataset demonstrates physics-based constraints that thermodynamics guarantees—making it excellent for showing how domain knowledge translates directly to model constraints.
Medical Insurance grounds constraints in actuarial science
The Medical Cost Personal dataset (1,338 rows, 7 features) predicts individual healthcare charges Dataquest with relationships that actuaries have validated for decades.
Source: https://www.kaggle.com/datasets/mirichoi0218/insurance
Key monotonic variables:

age (+1): Regression coefficient ~$255 per year (p<0.001)—older individuals have higher health expenses Towards Data Science
bmi (+1): Coefficient ~$321 per unit—obesity-related conditions increase costs Towards Data Science
children (+1): Each dependent adds ~$430 in potential medical expenses Towards Data Science
smoker (+1 when encoded): The strongest effect at ~$23,500 higher costs Towards Data Science

Insurance pricing is a production domain where monotonic constraints aren't just nice-to-have—they're regulatory requirements. This dataset demonstrates business-justified constraints with documented statistical significance.
Concrete Compressive Strength brings engineering rigor
The UCI Concrete dataset (1,030 rows, 8 features) predicts compressive strength (MPa) UCI Machine Learning Repository with one particularly strong monotonic relationship: curing age.
Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
Key monotonic variables:

Age (+1): Concrete follows logarithmic strength gain through hydration—28-day curing is standard because strength continues increasing
Cement (+1): More portland cement provides more binding material
Superplasticizer (+1): Reduces water-cement ratio, increasing strength

Civil engineering research confirms these relationships. The age variable has the strongest domain validation—strength = A*ln(age) + B is a known formula in concrete science. Taylor & Francis
Ames Housing supports many simultaneous constraints
For demonstrating multiple constraints simultaneously, the Ames Housing dataset (2,930 rows, 79 features) provides the richest feature set. GitHub
Source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
Key monotonic variables:
VariableDirectionRationaleGrLivArea+1Larger living area increases valueTotalBsmtSF+1More basement space adds utilityOverallQual+11-10 quality rating directly impacts priceGarageCars+1More garage capacity is desirableFullBath+1Additional bathrooms add valueYearBuilt+1Newer construction commands premiumsFireplaces+1Amenity that adds value
The 79 features include 36 numerical variables, many with clear monotonic expectations. github This enables demonstrating constraint selection—deciding which features warrant constraints versus those with ambiguous relationships.
Quick reference for implementation
DatasetSourceRowsFeaturesBest Monotonic VariablesConstraint DirectionsCalifornia Housingsklearn20,6408MedInc, AveRoomsBoth positiveAuto MPGUCI3987weight, displacement, model_yearMostly negativeMedical InsuranceKaggle1,3387age, bmi, smokerAll positiveConcrete StrengthUCI1,0308Age, CementBoth positiveAmes HousingKaggle2,93079GrLivArea, OverallQual, YearBuiltAll positive
