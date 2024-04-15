# Cost Models

A suite of open source parametric cost models.

# Constructive Systems Engineering Cost Model (COSYSMO)

Use COSYSMO per the following to provide size and cost factor inputs, compute effort and decompose it by activity.

```python
# test case
size_drivers = {
    "System Requirements": {
        "Easy": 12,
        "Nominal": 50,
        "Difficult": 13
    },
    "Interfaces": {
        "Easy": 2,
        "Nominal": 8,
        "Difficult": 6
    },
    "Critical Algorithms": {
        "Easy": 12,
        "Nominal": 14,
        "Difficult": 11
    },
    "Operational Scenarios": {
        "Easy": 2,
        "Nominal": 4,
        "Difficult": 3
    }
}

cost_factors = {
    "Requirements Understanding": "High",
    "Technology Risk": "High",
    "Process Capability": "High"
}

effort = cosysmo(size_drivers, cost_factors)
print(f'Effort = {effort:.1f} Person-Months')

phase_effort(effort)
```

```
Effort = 183.8 Person-Months
Activities                Effort    
-----------------------------------
Acquisition and Supply    12.86
Technical Management      31.24
System Design             55.13
Product Realization       27.56
Technical Evaluation      56.97
```
