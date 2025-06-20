# Discrete Event Modeling with se-lib – User's Guide

 ### April 28, 2025


<a id="Overview"></a>
## 1. Overview and Getting Started

This user's guide describes constructing and simulating discrete event models using the Systems Engineering Library (se-lib). The discrete event modeling capability allows for simulation of systems described as networks consisting of nodes that are connected by paths for entity flow.  Simulation is event-driven with tracking of individual entities as they pass through the network of process nodes.

To begin, import se-lib and initialize a discrete event model.

```python
from selib import *
init_de_model()
```

This call prepares the model environment, sets up the simulation clock, and initializes the discrete event network structure.

---
<a id="Defining Model Components"></a>
## 2. Defining Model Components

A system model is described by defining nodes for sources, servers with queues, delays, and terminations. Entity path logic between nodes is also specified.

### 2.1 Sources and Entities

 Source nodes generate entities with desired timings. They are system entry points for each entity that represents a task, product, or unit that flows through the system.

```python
add_source(
    name="Arrival",
    entity_name="Car",
    num_entities=100,
    connections={'Charger': 1},
    interarrival_time="random.uniform(1, 5)"
)
```

The source generates entities of the given name, spaced by a specified interarrival time. The entity names and their sequential IDs can be used in model logic.
Connections define which node(s) the entity proceeds to with relative probabilities.  

In the above example entities only connect to a single next node ```Charger``` with a 100% probability.  An example for multiple connections to two parallel servers with 50% probability each would be:

```python
    connections={'Charger1': .5, 'Charger2': .5},
```

Note the path weights are relative to each other. Thus any value for a single path would suffice as 100%, and any two equal values would allocate 50% probability.  For example, ```connections={'Charger1': 50, 'Charger2': 50}``` or even ```connections={'Charger1': 1, 'Charger2': 1}``` will have the same probabilistc effect.

### 2.2 Servers

Servers represent resources with constrained capacity, where queues form before the servers.

```python
add_server(
    name="Charger",
    connections={"Exit": 1.0},
    service_time="random.uniform(2, 5)",
    capacity=1
)
```

Servers queue entities and delay them based on the service time expression. The capacity controls how many can be served concurrently.

### 2.3 Delays

Delays correspond to the travel times of entities
from one node to the next.  They hold entities for a fixed or variable time without queueing.

```python
add_delay(
    name="Payment Approval",
    connections={"Charger": 1.0},
    delay_time="2"
)
```

### 2.4 Termination Nodes

Terminate nodes represent the path ends for entities where they are removed from the system.

```python
add_terminate(name="Exit")
```

---

## 3 Common Processes and Modeling Patterns

The following examples illustrate common processes and discrete event modeling structures using se-lib.

###  3.1 Single Server Queueing System

This model demonstrates a basic single-server queue. Entities are generated at regular intervals and may experience waiting delays due to limited capacity at the server. The queue accumulates when entity arrivals exceed the server’s ability to process them.

The output from this will display the model diagram and default run output.

```python
import random as random

init_de_model()
add_source(
    name="Arrival",
    entity_name="Car",
    num_entities=100,
    connections={'Charger': 1},
    interarrival_time="random.uniform(1, 5)"
)
add_server(
    name="Charger",
    connections={"Leave": 1.0},
    service_time="random.uniform(1, 4)",
    capacity=1
)
add_terminate(name="Leave")
draw_model()
run_model()
```

![single_server_car_charger](https://github.com/se-lib/se-lib/raw/main/docs/figures/single_server_car_charger.svg)

```
2.0555141567824218: Car 1 entered from Arrival
2.0555141567824218: Car 1 Arrival -> Charger
2.0555141567824218: Car 1 requesting Charger resource
2.0555141567824218: Car 1 granted Charger resource waiting time 0.0
3.9530893974237076: Car 2 entered from Arrival
3.9530893974237076: Car 2 Arrival -> Charger
3.9530893974237076: Car 2 requesting Charger resource
5.885350790323626: Car 1 completed using Charger resource with service time 3.8298366335412037
5.885350790323626: Car 1 Charger -> Leave
5.885350790323626: Car 1 leaving system at Leave
5.885350790323626: Car 2 granted Charger resource waiting time 1.9322613928999184
...
```

### 3.2 Path Delay

Entities experience a fixed or variable delay before entering a node.

```python
init_de_model()
add_source("Source", "Entity", 10, {"Delay": 1.0}, "1")
add_delay("Delay", {"Server": 1.0}, "2")
add_server("Server", {"Terminate": 1.0}, "3")
add_terminate("Terminate")
draw_model("delayed_path")
run_model()
```

![delayed_path](https://github.com/se-lib/se-lib/raw/main/docs/figures/delayed_path.svg)

```
1: Entity 1 entered from Source
1: Entity 1 Source -> Delay
2: Entity 2 entered from Source
2: Entity 2 Source -> Delay
3: Entity 3 entered from Source
3: Entity 3 Source -> Delay
```

In this example the function calls do not include the parameter input names. This optional syntax shortcut requires that the inputs are in the standard calling order.

### 3.3 Probabilistic Routing and Parallel Servers

This model contains nondeterministic routing to parallel servers. After passing through a central node, entities are dispatched to downstream paths with user-defined probabilities. The relative frequencies of path usage are reflected in the defined routing weights.

```python
init_de_model()
add_source("Source", "Entity", 10, {"Router": 1.0}, "1")
add_server("Router", {"Server A": 0.6, "Server B": 0.4}, "1")
add_server("Server A", {"Sink": 1.0}, "2")
add_server("Server B", {"Sink": 1.0}, "2")
add_terminate("Sink")
draw_model("path_routing")
run_model()
```
![path_routing](https://github.com/se-lib/se-lib/raw/main/docs/figures/path_routing.svg)


```
1: Entity 1 entered from Source
1: Entity 1 Source -> Router
1: Entity 1 requesting Router resource
1: Entity 1 granted Router resource waiting time 0
2: Entity 2 entered from Source
2: Entity 2 Source -> Router
2: Entity 2 requesting Router resource
2: Entity 1 completed using Router resource with service time 1
2: Entity 1 Router -> Server B
...
```

### 3.4 Capacity-Constrained Server

This model demonstrates the effect of constrained concurrency. A limited-capacity server processes multiple entities in parallel, up to the defined maximum. It reflects real-world constraints such as limited personnel, tools, or workstations.

```python
init_de_model()
add_source("Source", "Entity", 10, {"SharedResource": 1.0}, "1")
add_server("SharedResource", {"Exit": 1.0}, "5", capacity=2)
add_terminate("Exit")
draw_model("capacity_2")
run_model()

```

---
## 4. Visualizing Model Diagrams

A diagram of the model structure can be rendered using the ```draw_model``` function.

```python
draw_model(filename="single_server_car_charger", format="svg")
```

![single_server_car_charger](https://github.com/se-lib/se-lib/raw/main/docs/figures/single_server_car_charger.svg)

This produces a Graphviz-based model layout showing sources, servers, delays, and terminations along with their connections. The default format is svg and may be set to other supported output types including, pdf, png and jpeg files.  The format parameter in this example can be omitted since svg is the default.

---

<a id="running"></a>
## 5. Running a Simulation and Analyzing Output

Once all model components are defined, execute the simulation with ```run_model()```.  Entities are generated and tracked as they move through the network, and statistics are collected at each node. The default behavior is that verbose output will be displayed for each event and resulting model data will be shown.

```python
run_model()
```

```2.0555141567824218: Car 1 entered from Arrival
2.0555141567824218: Car 1 Arrival -> Charger
2.0555141567824218: Car 1 requesting Charger resource
2.0555141567824218: Car 1 granted Charger resource waiting time 0.0
3.9530893974237076: Car 2 entered from Arrival
...
```

The verbose output can be turned off with the ``verbose`` parameter.
```python
run_model(verbose=False)
```

### 5.1 Run Data
The ```run_model()``` function returns a tuple of model data and entity data for post-processing after execution. Model data includes node-level statistics for waiting times, service durations, resource utilization.  Entity data includes nodes visited, arrival and departure times for each entity ID.

The results are returned as a tuple and can be captured in dictionaries by naming them:

```python
model_data, entity_data = run_model()
```
These dictionaries can be examined, plotted, or exported depending on analysis needs.

```
print(model_data)
```

```
{'incoming_cars': {'type': 'source',
  'entity_name': 'Car',
  'num_entities': 50,
  'connections': {'charger': 0.7, 'impatient_cars': 0.3},
  'interarrival_time': 'np.random.exponential(5)',
  'arrivals': [1.8518275246814762,
   2.8837972333170807,
   3.2180364021037855,
   7.772066998169806,
...
'charger': {'type': 'server',
  'resource': <simpy.resources.resource.Resource at 0x7eecdb78d990>,
  'connections': {'payment': 1},
  'service_time': 'np.random.uniform(0, 16)',
  'waiting_times': [0.0,
   7.445018946036789,
   0.0,
   4.571154552428116,
   8.16315227890377,
   11.644193625248807,
   17.90869326875388,
   23.15034857718623,
   21.854958198126376,
   26.63050101289388,
   21.39761772342264,
   34.4534979183592,
   31.324463510453114,
   23.829072039820502,
   20.01940237435413,
   13.518156192205794,
   12.719070576090331,
...
```
Run statistics on nodes can be accessed in the output dictionary.  From the run output the waiting times for a server queue can be accessed with the following which provides a list of values.  For each node, se-lib captures the entity waiting times, service times, and utilization.

```
model_data['charger']['waiting_times']
```

```
[0.0,
 7.445018946036789,
 0.0,
 4.571154552428116,
...
```

Entity data contains event time data for each entity traversing through the network.

```
print(entity_data)
```

```
{1: {'arrival': 1.8518275246814762,
  'nodes': [('charger', 10.663055348140574),
   ('payment', 13.328187581559387),
   ('served_cars', 13.328187581559387)],
  'departure': 13.328187581559387},
 2: {'arrival': 2.8837972333170807,
  'nodes': [('impatient_cars', 2.8837972333170807)],
  'departure': 2.8837972333170807},
 3: {'arrival': 3.2180364021037855,
  'nodes': [('charger', 11.254405882251344),
   ('payment', 13.981810162773758),
   ('served_cars', 13.981810162773758)],
  'departure': 13.981810162773758},
...
```

### 5.2 Output Histograms

The run output can be sent for plotting histograms with ```plot_histogram()```.  It takes a lists which can be specified from the run output data.

```
plot_histogram(model_data['charger']['service_times'], xlabel="Charger Service Time")
```

<img src="https://github.com/se-lib/se-lib/raw/main/docs/figures/charger_service_times.png"
     alt="charger_service_times"
     width="400px">


---

## 7. Appendix A - Function Reference


*se-lib* provides two complementary interfaces for building and running discrete event models:

- **Procedural API** – A function-based interface that uses a single shared model instance behind the scenes.
- **Object-Oriented API** – A class-based interface that supports multiple model instances, encapsulated data, and better modularization.

The APIs offer the same modeling capabilities. The procedural interface is often more convenient for quick scripting but limited. The object-oriented interface is more scalable for larger simulations and programmatic reuse.

### Procedural API
These functions provide a procedural interface for discrete event modeling. A single instance of a model is implicit when using them.

### `init_de_model()`
Initializes a discrete event simulation model. Clears any existing structure and sets up a SimPy environment.

### `add_source(name, entity_name, num_entities, connections, interarrival_time)`
Defines a source node that generates entities with the given interarrival pattern.

### `add_server(name, connections, service_time, capacity=1)`
Adds a server with queueing behavior and optional parallel capacity.

### `add_delay(name, connections, delay_time)`
Adds a delay node that holds entities for a fixed or random duration.

### `add_terminate(name)`
Adds a termination node where entities exit the system.

### `run_model(verbose=True)`
Runs the model as a simulation. Returns model and entity data.

### `draw_model_diagram(filename=None, format="svg")`
Draws the model diagram using Graphviz.

These functions internally use a singleton `DiscreteEventModel` instance for compatibility.  The `DiscreteEventModel` class is described in the next section for the Object-Oriented API.

---

### `DiscreteEventModel` Class (Object-Oriented API)

A class for creating and simulating discrete event models.  For advanced use, `DiscreteEventModel` can be instantiated irectly. This allows the creation of multiple models, encapsulation of simulation logic in classes, and building of reusable components.

---

### `DiscreteEventModel()`
Creates a new empty discrete event model.

**Example**:
```python
model = DiscreteEventModel()
```

---

### `.add_server(name, connections, service_time, capacity=1)`
Adds a server node to the model.

**Args**:
- `name`: Name of the server.
- `connections`: Dictionary of next-node probabilities.
- `service_time`: Constant or expression for service duration.
- `capacity`: Number of resources (default is 1).

**Example**:
```python
model.add_server("Server1", {"Delay1": 1.0}, "random.expovariate(1.0)")
```

---

### `.add_delay(name, connections, delay_time)`
Adds a delay node to the model.

**Args**:
- `name`: Name of the delay.
- `connections`: Dictionary of next-node probabilities.
- `delay_time`: Duration expression.

---

### `.add_source(name, entity_name, num_entities, connections, interarrival_time)`
Adds a source node to the model.

**Args**:
- `name`: Name of the source node.
- `entity_name`: Label for generated entities.
- `num_entities`: Count of entities to generate.
- `connections`: Next-node probabilities.
- `interarrival_time`: Interval expression.

---

### `.add_terminate(name)`
Adds a termination node for entities.

**Args**:
- `name`: Node label.

---

### `.run(verbose=True)`
Runs the simulation and returns results.

**Args**:
- `verbose`: Print simulation events if True.

**Returns**:
- `network`, `entity_data`: Simulation structure and result logs.

**Example**:
```python
results = model.run(verbose=False)
```

---

### `.draw(filename=None, format='svg')`
Draws the model using Graphviz.

**Args**:
- `filename`: Optional name to save.
- `format`: Output format (e.g. 'svg', 'png').

---




