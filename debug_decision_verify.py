
import sys
sys.path.append('c:\\Users\\rabel\\Desktop\\se-lib')

from selib.decision import *
import pandas as pd

# 1. Define Criteria
criteria = [
    Criterion(name="Weight (kg)", weight=0.3, direction=CriterionDirection.MINIMIZE),
    Criterion(name="Endurance (min)", weight=0.5, direction=CriterionDirection.MAXIMIZE),
    Criterion(name="Cost ($)", weight=0.2, direction=CriterionDirection.MINIMIZE)
]

# 2. Define Alternatives
alternatives = [
    Alternative(name="Li-Po Battery", scores={
        "Weight (kg)": 2.5,
        "Endurance (min)": 25,
        "Cost ($)": 300
    }),
    Alternative(name="Hydrogen Fuel Cell", scores={
        "Weight (kg)": 3.0,
        "Endurance (min)": 60,
        "Cost ($)": 1500
    }),
    Alternative(name="Gasoline Engine", scores={
        "Weight (kg)": 4.0,
        "Endurance (min)": 90,
        "Cost ($)": 800
    }),
    Alternative(name="Hybrid", scores={
        "Weight (kg)": 3.5,
        "Endurance (min)": 75,
        "Cost ($)": 1200
    })
]

matrix = DecisionMatrix(criteria, alternatives)
print("Matrix DataFrame:")
print(matrix.to_dataframe())

# 3. Evaluation: Weighted Sum Model (WSM)
wsm = WeightedSumModel(matrix)
wsm_results = wsm.evaluate()
print("\nWSM Rankings:")
print(wsm_results)

# 4. Evaluation: TOPSIS
topsis = TOPSIS(matrix)
topsis_results = topsis.evaluate()
print("\nTOPSIS Rankings:")
print(topsis_results)

# 5. Evaluation: Pugh Matrix
pugh = PughMatrix(matrix, baseline_name="Li-Po Battery")
pugh_results = pugh.evaluate()
print("\nPugh Matrix Results:")
print(pugh_results)

print("\nVerification Script Completed Successfully.")
