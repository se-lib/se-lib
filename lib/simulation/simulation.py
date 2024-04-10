"""
se-lib Version .26.7

Copyright (c) 2022-2023 Ray Madachy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import graphviz
import textwrap
import os
import sys
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

from copy import deepcopy

import simpy
import random
import numpy as np

online = False

# system dynamics simulation functions

def init_sd_model(start, stop, dt):
  """
  Instantiates a system dynamics model for simulation
  """
  global xmile_header, model, model_specs, model_dict, model_type
  model_type = "continuous"
  xmile_header = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0">
        <header>
            <vendor>Ray Madachy</vendor>
            <name>Battle Simulator</name>
            <options>
                <uses_outputs/>
            </options>
            <product version="1.0">PyML .20 dev</product>
        </header>"""

  model_specs = f"""
        <sim_specs>
                <stop>{stop}</stop>
                <start>{start}</start>
                <dt>{dt}</dt>
        </sim_specs>"""
  model = ""
  build_model()
  model_dict={'stocks': {}, 'flows': {}, 'auxiliaries': {}}

def build_model():
  global xmile_string
  xmile_closing = """
    </xmile>
    """
  model_string = """
        <model>
            <variables>""" + f"{model}" + """
            </variables>
        </model>"""
  xmile_string = xmile_header + model_specs + model_string + xmile_closing
  with open('test.xmile', 'w') as f:
    f.write(xmile_string)
    
def add_stock(name, initial, inflows=[], outflows=[]):
  """
  Adds a stock to the model
    
  Parameters
  ----------
  name: str
    The name of the stock 
  initial: float
    Initial value of stock at start of simulation
  inflows: list of float
    The names of the inflows to the stock
  outflows: list of float
    The names of the outflows to the stock
  """
  global model
  inflow_string, outflow_string = "", ""
  for flow in inflows:
    inflow_string += f"""<inflow>"{flow}"</inflow>"""
  for flow in outflows:
    outflow_string += f"""<outflow>"{flow}"</outflow>"""
  model += f"""
                <stock name="{name}">
                    <doc>{name}</doc>
                    {inflow_string}
                    {outflow_string}
                    <eqn>{initial}</eqn>
                </stock>"""
  build_model()
  model_dict['stocks'][name]={'inflows': inflows, 'outflows': outflows}
  
def add_auxiliary(name, equation, inputs=[]):
  """
  Adds auxiliary equation or constant to the model

  Parameters
  ----------
  name: str
    The name of the auxiliary 
  equation: str
    Equation for the auxiliary using other named model variables
  inputs: list
    Optional list of variable input names used to draw model diagram 
  """
  if "random()" in str(equation): equation = convert_random_to_xmile(equation) 
  if "random.uniform(" in str(equation): equation = convert_random_to_xmile(equation)
  if "RANDOM" in str(equation): equation = equation.replace("RANDOM", '(GET_TIME_VALUE(0,0,0) + .00001) / (GET_TIME_VALUE(0,0,0) + .00001) * RANDOM')
  global model
  model += f"""
                <aux name="{name}">
                    <doc>{name}</doc>
                    <eqn>{equation}</eqn>
                </aux>"""
  build_model()
  model_dict['auxiliaries'][name]={'equation': equation, 'inputs': inputs}
    
def add_flow(name, equation, inputs=[]):
	"""
	Adds a flow to the model

	Parameters
	----------
	name: str
	  The name of the flow 
	equation: str
	  Equation for the flow using other named model variables
	inputs: list
	  Optional list of variable input names used to draw model diagram 
	"""
	if "random()" in str(equation): equation = convert_random_to_xmile(equation) 
	if "random.uniform(" in str(equation): equation = convert_random_to_xmile(equation)
	if "RANDOM" in str(equation): equation = equation.replace("RANDOM", '(GET_TIME_VALUE(0,0,0) + .00001) / (GET_TIME_VALUE(0,0,0) + .00001) * RANDOM')
	global model
	model += f"""
				<flow name="{name}">
					<doc>{name}</doc>
					<eqn>{equation}</eqn>
				</flow>"""
	build_model()
	model_dict['flows'][name]={'equation': equation, 'inputs': inputs}
	

def convert_random_to_xmile(equation):
  equation = equation.replace("random(", "RANDOM_0_1(")
  equation = equation.replace("random.uniform(", "RANDOM_UNIFORM(")
  return(equation)


def plot_graph(*outputs):
	"""
	displays matplotlib graph for each model variable

	Parameters
	----------
	variables: str or list
		comma separated variable name(s) or lists of variable names to plot on single graphs

	Returns
	----------
	matplotlib graph
	"""
	#print (outputs, len(outputs))
	for var in outputs:
		label_string=str(var)
		if type(var) == list:
			label_string=str(var[0])
			for count, element in enumerate(var):
				if count > 0: label_string += ", "+element
		fig, axis = plt.subplots(figsize=(6, 4))
		label_string = textwrap.fill(label_string, width=40)
		axis.set(xlabel='Time', ylabel=label_string)
		if type(var) == list:
			axis.plot(output.index, output[var].values, label=var)
			axis.legend(loc="best", )
		else: 
			axis.plot(output.index, output[var].values)
		plt.show()
  
def save_graph(*outputs, filename="graph.png"):
  """
  save graph to file

  Parameters
  ----------
  variables: variable name or list of variable names to plot on graph
  filename: file name with format extension
  """
  for var in outputs:
    label_string=str(var)
    if type(var) == list:
        label_string=str(var[0])
        for count, element in enumerate(var):
            if count > 0: label_string += ", "+element
    fig, axis = plt.subplots(figsize=(6, 4))
    label_string = textwrap.fill(label_string, width=40)
    axis.set(xlabel='Time', ylabel=label_string)
    if type(var) == list:
        axis.plot(output.index, output[var].values, label=var)
        axis.legend(loc="best", )
    else: 
        axis.plot(output.index, output[var].values)
    plt.savefig(filename)
            
def run_model(verbose=True):
	"""
	Executes the current model
    
	Returns
	----------
	If continuous, returns 1) Pandas dataframe containing run outputs for each variable each timestep and 2) model dictionary.
	If discrete, returns 1) network dictionary with run statistics and 2) entity run data
	"""
	verbose = verbose
	if (model_type == "continuous"): return(run_sd_model())
	if (model_type == "discrete"): return(run_de_model(verbose))

def run_sd_model():
    """
    Executes the model
    
    Returns
    ----------
    Pandas dataframe containing run outputs for each variable each timestep
    """
    import pysd
    global output
    global model
    model = pysd.read_xmile('./test.xmile')
    output = model.run(progress=False)
    return (output, model_dict)
    
def set_logical_run_time(condition):
    """
    Enables a run time to be measured based on a logical condition for when the simulation should be run (like a while statement).  The logical end time will be available from the 'get_logical_end_time()' function in lieu of the fixed end time for a simulation. 
    """
    add_flow("time_flow", 'if_then_else('+str(condition)+', 1, 0)')
    #add_flow("time_flow", 'if_then_else(not '+str(condition)+', 1, 0)')
    #add_flow("time_flow", 'if_then_else('+condition+', 0, 1)')
    add_stock("logical_end_time", 0, inflows=["time_flow"])
    
def get_logical_end_time():
    """
    Returns the logical end time as specified in a previous 'set_logical_run_time()' function call, in lieu of the fixed end time for a simulation. 
    
    Returns
    ----------
    logical_end_time: float
        end time when the 'set_logical_run_time()'' condition expires
    """
    return (get_final_value("logical_end_time"))  
    
def get_final_value(variable):
    return (output[variable][model['FINAL TIME']]) 
	
	
	
def draw_sd_model(filename=None, format='svg'):
    system_dynamics_dict = model_dict
    graph = graphviz.Digraph(engine='dot', filename=filename, format=format)
    graph.attr(rankdir='LR', size='10,8', splines='spline',)
    graph.attr('node', fontname="arial", fontcolor='blue', color='invis', fontsize='10')
    #graph.attr('edge',  minlen='1')
    
    
    with graph.subgraph(name='cluster_flowchain') as c:
        # Add stocks as boxes
        for stock_name in system_dynamics_dict['stocks']:
            graph.node(stock_name, shape='box', color='blue', )
        
        # Add flows as circles
        for flow_name in system_dynamics_dict['flows']:
            graph.node(flow_name, shape='circle', color='blue', width='.2', fixedsize="true" , label=f"\n\n{flow_name}")
        
    # Add auxiliaries as circles
    for aux_name in system_dynamics_dict['auxiliaries']:
        graph.node(aux_name, shape='circle', color='blue', width='.2', fixedsize="true", label=f"\n\n{aux_name}")
    
    # Add edges from inflows to stocks
    for stock_name, stock_dict in system_dynamics_dict['stocks'].items():
        for inflow_name in stock_dict['inflows']:
            graph.edge(inflow_name, stock_name, color="blue:blue", arrowhead="onormal")
            if inflow_name not in stock_dict['outflows']:
                graph.node(f'{inflow_name}_source', width=".01", fontsize='14', fixedsize="true", label="✽")
                graph.edge(f'{inflow_name}_source', inflow_name, tailclip="true", color="blue:blue", arrowhead="none")
    
    # Add edges from stocks to outflows
    for stock_name, stock_dict in system_dynamics_dict['stocks'].items():
        for outflow_name in stock_dict['outflows']:
            graph.edge(stock_name, outflow_name, color="blue:blue", arrowhead="none")
            if outflow_name not in stock_dict['inflows']:
                graph.node(f'{outflow_name}_sink', width=".01", fontsize='14', fixedsize="true", label="✽")
                graph.edge(outflow_name, f'{outflow_name}_sink', headclip="true", color="blue:blue", arrowhead="onormal")

    # Add edges from variable inputs to flows
    for flow_name, flow_dict in system_dynamics_dict['flows'].items():
        for input_name in flow_dict['inputs']:
            graph.edge(input_name, flow_name, color="red", arrowhead="normal", constraint='false')
 
           
    # Add edges from variable inputs to auxiliaries
    for aux_name, aux_dict in system_dynamics_dict['auxiliaries'].items():
        for input_name in aux_dict['inputs']:
            graph.edge(input_name, aux_name, color="red", arrowhead="normal", constraint='false')
    
    if filename is not None:
        graph.render()  # render and save file, clean up temporary dot source file (no extension) after successful rendering with (cleanup=True) doesn't work on windows "permission denied"
    return graph
	
	
def draw_model_diagram(filename=None, format="svg"):
    """
    Draw a diagram of the current model. 

    Parameters
    ----------
    filename : string, optional
        A filename for the output not including a filename extension. The extension will specified by the format parameter.
    format : string, optional
        The file format of the graphic output. Note that bitmap formats (png, bmp, or jpeg) will not be as sharp as the default svg vector format and most particularly when magnified.

    Returns
    -------
    g : graph object view
        Save the graph source code to file, and open the rendered result in its default viewing application. se-lib calls the Graphviz API for this.

    """
    filename = filename
    format = format
    if model_type == "continuous": return(draw_sd_model(filename, format))
    if model_type == "discrete": return(draw_discrete_model_diagram(filename, format, engine='dot'))	
	
# Discrete Event Modeling 

# define network dictionary of dictionaries
network = {}

run_specs = {}
entity_num = 0
entity_data = {}

def init_de_model():
    """
    Instantiates a discrete event model for simulation
    """
    global env, entity_num, entity_data, run_specs, model_type
    network.clear()
    run_specs.clear()
    entity_num = 0
    entity_data.clear()
    
    # create simulation environment
    env = simpy.Environment()
    model_type = "discrete"

def add_server(name, connections, service_time, capacity=1):
    """
    Add a server to a discrete event model. 

    Parameters
    ----------
    name: string
    	A name for the server.
    connections: dictionary
    	A dictionary of the node connections after the server.  The node names are the keys and the values are the relative probabilities of traversing the connection.
    capacity: integer
    	The number of resource usage slots in the server
    """
    network[name] = {
        'type': 'server',
        'resource': simpy.Resource(env, capacity=capacity),
        'connections': connections,
        'service_time': service_time,
        'waiting_times': [],
        'service_times': [],
        'capacity': capacity,
        'resource_busy_time': 0,
        'resource_utilization': 0
    }


def add_delay(name, connections, delay_time):
    """
    Add a delay to a discrete event model. 

    Parameters
    ----------
    name: string
    	A name for the delay.
    connections: dictionary
    	A dictionary of the node connections after the delay.  The node names are the keys and the values are the relative probabilities of traversing the connections.
    delay_time: float
    	The time delay for entities to traverse.  May be a constant or random function.
    """
    network[name] = {
        'type': 'delay',
        'connections': connections,
        'delay_time': delay_time,
        'delay_times': [],
    }


def add_source(name, entity_name, num_entities, connections, interarrival_time):
    """
    Add a source node to a discrete event model to generate entities. 

    Parameters
    ----------
    name: string
    	A name for the source.
    entity_name: string
    	A name for the type of entity being generated.
    num_entities: integer
    	Number of entities to generated.
    connections: dictionary
    	A dictionary of the node connections after the source.  The node names are the keys and the values are the relative probabilities of traversing the connections.
    interarrival_time: string
    	The time between entity arrrivals into the system.  The string may enclose a constant, random function or logical expression to be evaluated.
    """

    network[name] = {
        'type': 'source',
        'entity_name': entity_name,
        'num_entities': num_entities,
        'connections': connections,
        'interarrival_time': interarrival_time,
        'arrivals': []
    }
    run_specs['interarrival_time'] = interarrival_time
    run_specs['num_entities'] = num_entities
    run_specs['entity_name'] = entity_name
    run_specs['source'] = name


def add_terminate(name):
    """
    Add a terminate node to a discrete event model for entities leaving the system. 

    Parameters
    ----------
    name: string
    	A name for the terminate.
    """
    network[name] = {
        'type': 'terminate', 
        'connections': {}}


# define processes for each entity
def process_initial_arrival(env, arrival_time, node_name, entity_num, entity_name):
    yield env.timeout(arrival_time)
    if run_specs['verbose']: print(f"{env.now}: {entity_name} {entity_num} entered from {run_specs['source']}")
    env.process(process_node(env, node_name, entity_num, entity_name))

    # define processes for each node
def process_node(env, node_name, entity_num, entity_name):
    # process resource usage

    if network[node_name]['type'] == 'delay':
        delay_time = eval(network[node_name]['delay_time'])
        yield env.timeout(delay_time)
        network[node_name]['delay_times'].append(delay_time)
        if run_specs['verbose']: print(f"{env.now}: {entity_name} {entity_num} delayed {delay_time} at {node_name}")
        entity_data[entity_num]['nodes'].append((node_name, env.now))


    if network[node_name]['type'] == 'server':
        with network[node_name]['resource'].request() as req:
            if run_specs['verbose']: print(f"{env.now}: {entity_name} {entity_num} requesting {node_name} resource ")
            this_arrival_time = env.now
            #entity_data[entity_num]['nodes'].append((node_name, env.now))
            yield req

            waiting_time = env.now - this_arrival_time
            # collect waiting times
            network[node_name]['waiting_times'].append(waiting_time)
            if run_specs['verbose']: print(f"{env.now}: {entity_name} {entity_num} granted {node_name} resource waiting time {waiting_time}")
            service_time = eval(network[node_name]['service_time'])
            yield env.timeout(service_time)

            # collect service times
            network[node_name]['service_times'].append(service_time)
            entity_data[entity_num]['nodes'].append((node_name, env.now))
            network[node_name]['resource_busy_time'] += service_time
            network[node_name]['resource_utilization'] = network[node_name]['resource_busy_time']/env.now/network[node_name]['capacity'] 
            if run_specs['verbose']: print(f"{env.now}: {entity_name} {entity_num} completed using {node_name} resource with service time {service_time}")

    if network[node_name]['type'] == 'terminate':
        if run_specs['verbose']: print(f"{env.now}: {entity_name} {entity_num} leaving system at {node_name} ")
        entity_data[entity_num]['nodes'].append((node_name, env.now))
        entity_data[entity_num]['departure'] = env.now


            # process arrivals and connections
    if len(network[node_name]['connections']) > 0:
        weights = tuple(network[node_name]['connections'].values())
        #print(weights)
        connection, probability = random.choice(list(network[node_name]['connections'].items()))
        connection = random.choices(list(network[node_name]['connections'].keys()), weights, k=1)[0]
        if run_specs['verbose']: print(f"{env.now}: {entity_name} {entity_num} {node_name} -> {connection}")
        env.process(process_node(env, connection, entity_num, entity_name))


def run_de_model(verbose=True):
    """
    Executes the current model
    
    Returns:
    ----------
    Simulation output
    """
    #global verbose
    verbose = verbose
    run_specs['verbose'] = verbose
    for key, value in network.items():
        if value.get('type') == 'source':
            source_name = key
    arrival_time = 0
    for entity_num in range(run_specs['num_entities']):
        #yield env.timeout(1)
        arrival_time += eval(run_specs['interarrival_time'])
        network[source_name]['arrivals'].append(arrival_time)
        entity_num += 1

        entity_data[entity_num] = {'arrival': arrival_time, 'nodes': [],'departure': None}
        env.process(
            process_initial_arrival(env, arrival_time, source_name,
                                    entity_num, run_specs['entity_name']))  #+str(entity)

    # start simulation at first node
    #env.process(process_node(env, 'node1'))
    env.run()
    return network, entity_data


def draw_discrete_model_diagram(filename=None, format='svg', engine='dot'):
    global graph
    node_attr = {
        'color': 'black',
        'fontsize': '11',
        'fontname': 'arial',
        'shape': 'none'
    }  # 'fontname': 'arial',
    graph = graphviz.Digraph('G',
                             node_attr=node_attr,
                             filename=filename,
                             format=format,
                             engine=engine)
    graph.attr(rankdir='LR', ranksep='.7')
    for node_name in network:
        if network[node_name]['type'] == "source":
            graph.node(node_name, label=f'❚❚❱\n\{node_name}', shape='none')
        if network[node_name]['type'] == "terminate":
            graph.node(node_name, label=f'❱❚❚\n\{node_name}', shape='none')
        if network[node_name]['type'] == "server":
            graph.node(node_name, label=f'❚❚▣\n{node_name}', shape='none')
        if network[node_name]['type'] == "delay":
            graph.node(node_name, label=f'◷\n{node_name}', shape='none')
        for connection in network[node_name]['connections']:
            graph.edge(
                node_name,
                connection,
            )
    if filename is not None:
        graph.render(
        )  # render and save file, clean up temporary dot source file (no extension) after successful rendering with (cleanup=True) doesn't work on windows "permission denied"
    return graph
    
def plot_histogram(data, filename=None, xlabel= "Data"):
    """
    Plot a histogram for a dataset and optionally save to a file.

    Parameters
    ----------
    data: list
    	A list of the data values
    filename: string, optional
    	A name for the file 	
    xlabel: string , optional
    	A label for the x-axis

    Returns
    -------
    A Matplotlib histogram
    """
    kwargs = {'color': 'blue', 'rwidth': 0.9}
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(data, **kwargs)
    ax.set(xlabel=xlabel, ylabel='Frequency')
    ax.xaxis.labelmargin = -50
    ax.yaxis.labelmargin = 30
    plt.subplots_adjust(bottom=0.15,)
    if filename is not None: plt.savefig(filename)
    return fig
