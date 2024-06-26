def cosysmo(size_drivers, cost_factors):
    return .254 * eaf(cost_factors) *total_size(size_drivers)**1.06
	
def eaf(ratings_dict):
    eaf_value = 1.0
    
    for cost_factor, rating_values in cost_factors_dict.items():
        # Get the rating from the ratings_dict or default to 'Nominal' if not provided
        rating = ratings_dict.get(cost_factor, "Nominal")
        
        # Retrieve the effort multiplier for the rating
        multiplier = rating_values.get(rating, 1.0)  # default to 1.0 if the rating doesn't exist for some reason
        
        # Multiply the eaf_value by the multiplier
        eaf_value *= multiplier
        
    return eaf_value
	
cost_factors_dict = {
    "Requirements Understanding": {"Very_Low": 1.87, "Low": 1.37, "Nominal": 1.00, "High": 0.77, "Very_High": 0.60},
    "Architecture Understanding": {"Very_Low": 1.64, "Low": 1.28, "Nominal": 1.00, "High": 0.81, "Very_High": 0.65},
    "Level of Service Requirements": {"Very_Low": 0.62, "Low": 0.79, "Nominal": 1.00, "High": 1.36, "Very_High": 1.85},
    "Migration Complexity": {"Nominal": 1.00, "High": 1.25, "Very_High": 1.55, "Extra_High": 1.93},
    "Technology Risk": {"Very_Low": 0.67, "Low": 0.82, "Nominal": 1.00, "High": 1.32, "Very_High": 1.75},
    "Documentation": {"Very_Low": 0.78, "Low": 0.88, "Nominal": 1.00, "High": 1.13, "Very_High": 1.28},
    "Diversity of Installations/Platforms": {"Nominal": 1.00, "High": 1.23, "Very_High": 1.52, "Extra_High": 1.87},
    "Recursive Levels in Design": {"Very_Low": 0.76, "Low": 0.87, "Nominal": 1.00, "High": 1.21, "Very_High": 1.47},
    "Stakeholder Team Cohesion": {"Very_Low": 1.50, "Low": 1.22, "Nominal": 1.00, "High": 0.81, "Very_High": 0.65},
    "Personnel/Team Capability": {"Very_Low": 1.50, "Low": 1.22, "Nominal": 1.00, "High": 0.81, "Very_High": 0.65},
    "Personnel Experience/Continuity": {"Very_Low": 1.48, "Low": 1.22, "Nominal": 1.00, "High": 0.82, "Very_High": 0.67},
    "Process Capability": {"Very_Low": 1.47, "Low": 1.21, "Nominal": 1.00, "High": 0.88, "Very_High": 0.77, "Extra_High": 0.68},
    "Multisite Coordination": {"Very_Low": 1.39, "Low": 1.18, "Nominal": 1.00, "High": 0.90, "Very_High": 0.80, "Extra_High": 0.72},
    "Tool Support": {"Very_Low": 1.39, "Low": 1.18, "Nominal": 1.00, "High": 0.85, "Very_High": 0.72}
}

def phase_effort(effort):
    # Define the effort distribution percentages for each activity
    activity_percentages = {
        "Acquisition and Supply": 7/100,
        "Technical Management": 17/100,
        "System Design": 30/100,
        "Product Realization": 15/100,
        "Technical Evaluation": 31/100
    }
    
    # Calculate the sub-effort for each activity based on its percentage
    sub_efforts = {activity: effort * percentage for activity, percentage in activity_percentages.items()}
    
    # Print the table of the effort outputs
    print(f"{'Activities':<25} {'Effort':<10}")
    print("-" * 35)
    for activity, sub_effort in sub_efforts.items():
        print(f"{activity:<25} {sub_effort:.2f}")
    
    return sub_efforts
	
size_weights = {
    "System Requirements": {
        "Easy": 0.5,
        "Nominal": 1.00,
        "Difficult": 5.0
    },
    "Interfaces": {
        "Easy": 1.1,
        "Nominal": 2.8,
        "Difficult": 6.3
    },
    "Critical Algorithms": {
        "Easy": 2.2,
        "Nominal": 4.1,
        "Difficult": 11.5
    },
    "Operational Scenarios": {
        "Easy": 6.2,
        "Nominal": 14.4,
        "Difficult": 30
    }
}

def total_size(driver_counts):
    total = 0
    for driver, complexities in driver_counts.items():
        for complexity, count in complexities.items():
            total += count * size_weights[driver][complexity]
    return total
	
def compute_effort():
    size_drivers = {
        driver: {complexity: float(boxes[driver][complexity].get()) for complexity in complexities}
        for driver in size_weights
    }

    cost_factors_ratings = {
        factor: factor_comboboxes[factor].get()
        for factor in cost_factors_dict
    }

    effort = cosysmo(size_drivers, cost_factors_ratings)

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
