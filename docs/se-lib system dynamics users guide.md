# System Dynamics Modeling with se-lib – User's Guide

### April 17, 2025

## Table of Contents

- [1. Overview](#1-overview)
- [2. Getting Started with se-lib](#2-getting-started-with-se-lib)
- [3. Modeling Guidance and Tips](#3-modeling-guidance-and-tips)
- [4. Basic Modeling Patterns](#4-basic-modeling-patterns)
  - [4.1 Negative Feedback Loop](#41-negative-feedback-loop)
  - [4.2 Exponential Growth](#42-exponential-growth)
  - [4.3 Balancing Loop](#43-balancing-loop)
  - [4.4 Goal-Gap Structure](#44-goal-gap-structure)
  - [4.5 First-Order Delay](#45-first-order-delay)
  - [4.6 Second-Order Delay](#46-second-order-delay)
- [5. Displaying Output](#5-displaying-output)
- [6. Graphing Functions](#6-graphing-functions)
- [7. Random Number Functions](#7-random-number-functions)
- [8. Utility and Test Functions](#8-utility-and-test-functions)
- [9. Advanced Usage](#9-advanced-usage)
- [9.1 Extending a Model Instance](#9-advanced-usage)
- [10. Visual Gallery of Model Structure](#10-visual-gallery-of-model-structures)
- [11. Appendix A - Function Reference](#11-appendix-a---function-reference)

---

## 1. Overview

This manual provides an introduction and reference for modeling continuous systems using the *se-lib* Python package. se-lib supports the construction and simulation of system dynamics models using standard stock and flow structures using both procedural and object-oriented approaches. It adheres to the XMILE standard for system dynamics models and integrates with [PySD](https://pysd.readthedocs.io/) for simulation execution. Models are compatible with Vensim and iThink/Stella tools using XMILE and MDL formats for importing and exporting.

The object-oriented interface is built around the `SystemDynamicsModel` class, enabling encapsulation of models, concurrent instantiation, and extensions of the model class. Users may also use the original procedural API for quick prototyping or compatibility with existing scripts.  The procedural approach provides a more implicit modeling, but can be less flexible. 

The manual includes reusable modeling templates, core system dynamics patterns such as goal-gap control and balancing loops, recommended modeling practices, detailed API documentation, and visual representations of feedback logic. It is suitable for a wide range of applications in systems thinking, behavioral simulation, engineering analysis, and education.

Detailed function references and more examples are available online at http://selib.org/function_reference.html#system-dynamics.


---

## 2. Getting Started with SE-Lib

To install the latest version of se-lib, use pip:

```
pip install se-lib
```

A system model is described by defining the standard elements for stocks (levels), flows (rates),
and auxiliary constants or equations. Utility functions are available for equation formulation, data collection and displaying
output.

Model construction begins by importing se-lib and initializing a model with the required time parameters. Simulations operate over a user-defined time horizon and return data as pandas DataFrames, which can be used for analysis and visualization.

```python
from selib import *
init_sd_model(start=0, stop=30, dt=1)
```

Refer to Section 11 for a complete description of the modeling API.

---

## 3. Modeling Guidance and Tips

Modeling with se-lib requires attention to syntax, naming conventions, and logical structure. Names of model elements are specified as character strings and should not contain spaces; underscores (`_`) must be used to separate terms (e.g., `Production_Rate`). Variable names are case-sensitive and must exactly match those used in equations.

Equations are expressed as strings using standard arithmetic operators. Variables referenced in equations must already be defined within the model scope. The simulator does not check equations at entry time, so errors from undefined variables or misspellings will only become apparent at runtime.

Conditional statements can be expressed with logical constructs `or`, `and`, and `not` or the XMILE uppercase standards `OR`, `AND`, and `NOT`.  The `if_then_else(condition, value if true, value if false)` or uppercase `IF_THEN_ELSE(condition, value if true, value if false)` are available.

Random number functions such as `random()` and `random.uniform(a, b)` are supported and internally translated into XMILE-compliant expressions. These generate new values at each timestep and are useful for modeling uncertainty or variability in external influences.

Time in se-lib is dimensionless. Modelers must maintain consistency in time unit interpretation (e.g., using all variables in units of days, months, etc.). Inconsistent use of time units can lead to misleading or invalid results.

The `draw_model()` function is recommended for verifying model structure prior to simulation. This visual representation helps ensure the correctness of stock-flow relationships and the presence of intended feedback loops.

---

## 4. Basic Modeling Patterns

### 4.1 Negative Feedback Loop

This classic balancing structure adjusts a level toward a target over time. It works by calculating the gap between a goal and the current level, then feeding a corrective flow in the direction that reduces this gap. Over time, the level asymptotically approaches the goal value.

```python
init_sd_model(start=0, stop=10, dt=0.1)
add_stock("level", 50, inflows=["rate"])
add_auxiliary("goal", 100)
add_auxiliary("time_constant", 0.5)
add_flow("rate", "(goal - level) / time_constant")
run_model()
plot_graph("level")
```

### 4.2 Exponential Growth

This reinforcing loop demonstrates positive feedback. The flow increases as the stock increases, which in turn further amplifies the flow. This results in exponential behavior and is frequently seen in unchecked population growth, viral spread, or compound interest.

```python
init_sd_model(start=0, stop=10, dt=0.2)
add_stock("x", 1000, inflows=["dx"])
add_auxiliary("growth_factor", 0.2)
add_flow("dx", "x * growth_factor")
run_model()
plot_graph("x")
```

### 4.3 Balancing Loop

A balancing loop with inventory control adjusts production to maintain a desired stock level. When inventory falls below target, production increases; when it's above target, production slows. This loop exhibits goal-seeking behavior and helps stabilize system performance. External demand acts as an opposing outflow.

```python
init_sd_model(0, 20, 1)
add_stock("Inventory", 50, inflows=["Production"], outflows=["Demand"])
add_auxiliary("Target_Inventory", 100)
add_auxiliary("Adj_Time", 4)
add_flow("Production", "(Target_Inventory - Inventory) / Adj_Time")
add_auxiliary("Demand", 10)
add_flow("Demand", "Demand")
run_model()
plot_graph("Inventory")
```

### 4.4 Goal-Gap Structure

This control structure reduces the deviation between a system’s state and a set point. A feedback signal (difference between goal and current value) is amplified by a gain factor and used to drive the stock toward the goal.

```python
init_sd_model(0, 30, 1)
add_stock("Temperature", 20, inflows=["Heating"])
add_auxiliary("Set_Point", 22)
add_auxiliary("Gain", 1.5)
add_flow("Heating", "(Set_Point - Temperature) * Gain")
run_model()
plot_graph("Temperature")
```

### 4.5 First-Order Delay

First-order delays represent lagged responses where a stock adjusts toward an input with exponential smoothing. The greater the delay time, the slower the convergence.

```python
init_sd_model(0, 15, 1)
add_stock("Output", 0, inflows=["Flow"])
add_auxiliary("Input", 100)
add_auxiliary("Delay_Time", 5)
add_flow("Flow", "(Input - Output) / Delay_Time")
run_model()
plot_graph("Output")
```

### 4.6 Second-Order Delay

A second-order delay structure creates smoother responses by cascading two first-order delays. This is useful when modeling systems with inertia or buffer stages such as material transport, bureaucratic processes, or cognitive recognition lags.

```python
init_sd_model(0, 30, 1)
add_stock("Stage1", 0, inflows=["Flow1"], outflows=["Flow2"])
add_stock("Stage2", 0, inflows=["Flow2"])
add_auxiliary("Input", 100)
add_auxiliary("Delay", 5)
add_flow("Flow1", "(Input - Stage1) / Delay")
add_flow("Flow2", "(Stage1 - Stage2) / Delay")
run_model()
plot_graph(["Stage1", "Stage2"])
```


---

## 5. Displaying Output

Upon execution of a system dynamics model, se-lib returns simulation results as a `pandas.DataFrame` object. This output contains the time-indexed values of all defined stocks, flows, and auxiliaries. The DataFrame facilitates further analysis, visualization, or export to external tools.
The ```run_model()`` function will execute a simulation and display a Pandas dataframe of the output.  To execute a model and capture its output:

```python
results = run_model()
```

Each column in the DataFrame corresponds to a model variable, and each row represents a timestep, as defined by the start, stop, and `dt` parameters provided during model initialization.

To inspect specific variables or values at particular timesteps, standard pandas indexing may be used:

```python
results['Population']               # Full time series for Population
results['Births'][10]              # Value of Births at the 10th timestep
results[['Population', 'Births']] # Multiple variables
```

This structured output enables direct integration with plotting functions, statistical analysis routines, or export mechanisms. All graphing and saving operations described in Section 6 operate directly on this result structure.

---

## 6. Graphing Functions

se-lib provides built-in graphing capabilities to visualize model outputs directly from the simulation results. These graphs are generated using the `matplotlib` library and support both single-variable and multi-variable plots.

### Plotting to Screen

The `plot_graph()` function displays simulation results interactively. It supports multiple input formats: single variable names, separate arguments, or a list of variables.

```python
plot_graph("Population")
plot_graph("Births", "Deaths")
plot_graph(["Population", "Births", "Deaths"])
```

When a list is used, variables are plotted together on a shared axis. When passed as separate arguments, they are displayed on individual subplots.

### Saving to File

The `save_graph()` function saves output graphs to image files. The default format is PNG, and the format may be changed by modifying the filename extension.

```python
save_graph("Population", filename="population_plot.png")
save_graph(["Births", "Deaths"], filename="flow_graph.svg")
```

Saved graphs are rendered as separate figures and written to disk, which is useful for reporting or presentation workflows. As with plotting, graphing functions operate on the output returned by `run_model()`.

---

## 7. Random Number Functions

se-lib supports stochastic modeling using random number generation within auxiliary and flow equations. This allows the representation of uncertainty, noise, or nondeterministic behavior in the modeled system.  In equations for auxiliaries and rates, the random number functions supported are called
as if the following import has occurred: ```from random import random, random.uniform```.  

The random functions supported are:

- `random()`: returns a new float uniformly distributed between 0 and 1 at each timestep.
- `random.uniform(min, max)`: returns a new float uniformly distributed between `min` and `max` at each timestep.

These functions are automatically translated to XMILE-compatible expressions when the model is compiled for simulation.  For xmile format compatibility, the functions ```RANDOM_0_1``` and ```RANDOM_UNIFORM(min, max)``` are equivalent and also acceptable.

### Example

The following example defines an auxiliary variable using random noise:

```python
init_sd_model(start=0, stop=3, dt=.5)
add_auxiliary("random_parameter", "20*random()")
run_model()
```

```
( INITIAL TIME FINAL TIME TIME STEP SAVEPER random_parameter
0.0 0 3 0.5 0.5 0.087466
0.5 0 3 0.5 0.5 11.174929
1.0 0 3 0.5 0.5 13.506344
1.5 0 3 0.5 0.5 4.342211
2.0 0 3 0.5 0.5 5.449781
2.5 0 3 0.5 0.5 7.323554
3.0 0 3 0.5 0.5 17.220789,
```

Each timestep generates a new random value for `Stochastic_Input`, producing a non-repeating time series. It is important to note that random values are recomputed each time the model is run, and results may vary unless reproducibility measures (e.g., seeding) are implemented outside SE-Lib.

---

## 8. Utility and Test Functions
Standard test functions are available for pulse, ramp and step inputs as shown below. The full set
of [functions available in PySD](https://pysd.readthedocs.io/en/master/structure/xmile_translation.html#xmile-supported-functions) can be used but have not all been tested.
```python
init_sd_model(0, 10, 1)
add_stock("Level", 0, inflows=["Pulse", "Ramp"])
add_flow("Pulse", "pulse(100, 2)")
add_flow("Ramp", "ramp(3, 5)")
run_model()
plot_graph("Pulse", "Ramp", "Level")
```

---

## 9.  Advanced Usage

### 9.1 Extending a Model Instance

Create a new model that extends the population model by adding carrying capacity with the following.

```python
class PopulationWithCapacityModel(SystemDynamicsModel):
    def __init__(self, start, stop, dt, initial_population, carrying_capacity):
        super().__init__(start, stop, dt)
        self.add_stock("Population", initial=initial_population, inflows=["Births"], outflows=["Deaths"])
        self.add_auxiliary("BirthRate", equation=0.02)
        self.add_auxiliary("CarryingCapacity", equation=carrying_capacity)
        self.add_auxiliary("CrowdingEffect", equation="Population / CarryingCapacity")
        self.add_auxiliary("EffectiveGrowthRate", equation="BirthRate * (1 - CrowdingEffect)")
        self.add_flow("Births", equation="Population * EffectiveGrowthRate")
        self.add_auxiliary("DeathRate", equation=0.01)
        self.add_flow("Deaths", equation="Population * DeathRate") # Keep a simple death rate

# Create an instance of the extended model
population_capacity_model = PopulationWithCapacityModel(start=0, stop=200, dt=1, initial_population=50, carrying_capacity=500)
```

### 9.1 PySD Integration

SE-Lib is fully compatible with PySD. Any PySD function or supported XMILE structure may be used where appropriate. See [PySD Docs](https://pysd.readthedocs.io/) for advanced formulations.

---

## 10. Visual Gallery of Model Structures

Here are illustrative stock-flow diagrams created with `draw_model_diagram()`.

**Negative Feedback Loop**
```
  Goal →(+)→ Rate →(+)→ Level
                  ↑       ↓
              (-)←--------
```

**Exponential Growth**
```
    Growth_Factor
           ↓
    Stock (x) → Flow (dx) → x
```

**Goal-Gap Regulation**
```
  Set_Point → Error → Actuator → Stock
      ↑                         ↓
     -----------------------------
```

To render actual diagrams, use:
```python
draw_model_diagram("model_diagram")
```

See section 2 for code to generate these models.

---


## 11. Appendix A - Function Reference

The following functions and methods define the modeling API for se-lib. All functions can be used in either procedural or object-oriented workflows.

### `init_sd_model(start, stop, dt, stop_when=None)`
Initializes a new system dynamics model over the specified simulation time frame.

**Args**:
- `start` (float): Start time of the simulation.
- `stop` (float): End time of the simulation.
- `dt` (float): Time step for numerical integration.
- `stop_when` (float): Logical condition using model variables to stop the simulation.

**Returns**:
- SystemDynamicsModel instance

**Example**:
```python
init_sd_model(start=0, stop=50, dt=1.0)
```

**Examples with stop_when**:
```python
init_sd_model(start=0, stop=50, dt=1.0, stop_when("stock1 >= 100")
```

```python
init_sd_model(start=0, stop=50, dt=1.0, stop_when("stock1 >= 100 or stock2 >=80")
```
---

### `add_stock(name, initial, inflows=[], outflows=[])`
Adds a stock variable (level) to the system dynamics model.

**Args**:
- `name` (str): Name of the stock.
- `initial` (float): Initial value of the stock.
- `inflows` (list, optional): Names of inflow variables affecting this stock.
- `outflows` (list, optional): Names of outflow variables affecting this stock.

**Returns**:
- None

**Example**:
```python
add_stock("Population", initial=100, inflows=["Births"], outflows=["Deaths"])
```

---

### `add_flow(name, equation)`
Defines a flow (rate of change) for use in stock updates.

**Args**:
- `name` (str): Name of the flow.
- `equation` (str): Equation as a string to compute the flow value.

**Returns**:
- None

**Example**:
```python
add_flow("Births", "Birth_Rate * Population")
```

---

### `add_auxiliary(name, equation)`
Defines an auxiliary variable or constant used in the model.

**Args**:
- `name` (str): Name of the auxiliary variable.
- `equation` (str): String representation of the equation or constant value.

**Returns**:
- None

**Example**:
```python
add_auxiliary("Birth_Rate", "0.02")
```

---

### `add_auxiliary_lookup(name, input, xpoints, ypoints)`
Creates a lookup function for a nonlinear relationship.

**Args**:
- `name` (str): Name of the auxiliary variable.
- `input` (str): Name of the input variable used in the lookup.
- `xpoints` (list): List of input values (x-axis).
- `ypoints` (list): List of output values (y-axis).

**Returns**:
- None

**Example**:
```python
add_auxiliary_lookup("Saturation", "Population", [0, 1000, 2000], [1, 0.5, 0.1])
```

---

### `plot_graph(*outputs)`
Plots model results using `matplotlib`.

**Args**:
- `*outputs` (str or list): Names of variables to plot (e.g., "Population", ["Births", "Deaths"]).

**Returns**:
- None

**Example**:
```python
plot_graph("Population")
plot_graph(["Births", "Deaths"])
```

---

### `save_graph(*outputs, filename='graph.png')`
Saves a plot of the simulation

### draw_sd_model(filename=None, format='svg')
Generates and renders a visual system dynamics model diagram using Graphviz.

### get_stop_time()
Returns the logical end time based on a stop_when condition.

### get_model_structure()
Returns a dictionary of model components (stocks, flows, auxiliaries).


