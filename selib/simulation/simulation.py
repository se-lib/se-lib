"""
se-lib Version .41

Copyright (c) 2022-2025 se-lib Development Team

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
import html
import xml.etree.ElementTree as ET
import re
import platform
import tempfile
import subprocess
from typing import Dict, List, Union, Optional, Callable, Any, Tuple
from IPython.display import SVG, Image, display

# Configure matplotlib defaults
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

import simpy
import random
import numpy as np

online = False

####################### SYSTEM DYNAMICS ###############################

class SystemDynamicsModel:
    """
    A class representing a System Dynamics model.

    This class provides methods to define, simulate and visualize system dynamics models,
    including stocks, flows, and auxiliaries.
    """

    def __init__(self, start: float, stop: float, dt: float, stop_when: Optional[str] = None):
        """
        Initialize a system dynamics model.

        Parameters
        ----------
        start : float
            Start time of the simulation
        stop : float
            Stop time of the simulation
        dt : float
            Time step
        stop_when : str, optional
            Optional logical condition to embed in <stop_when> in the XMILE header
        """
        self.start = start
        self.stop = stop
        self.dt = dt
        self.stop_when = stop_when

        self.model_dict = {'stocks': {}, 'flows': {}, 'auxiliaries': {}}
        self.model = ""
        self.xmile_string = ""
        self.run_output = None

        # Create XMILE header
        escaped_stop = html.escape(stop_when) if stop_when else ""
        stop_tag = f"    <stop_when>{escaped_stop}</stop_when>\n" if stop_when else ""

        self.xmile_header = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0">
            <header>
                <vendor>se-lib</vendor>
                <name></name>
                <options>
                    <uses_outputs/>
                </options>
                <product version=".35">se-lib</product>
                {stop_tag}
            </header>"""

        self.model_specs = f"""
            <sim_specs>
                    <stop>{stop}</stop>
                    <start>{start}</start>
                    <dt>{dt}</dt>
            </sim_specs>"""

        self.build_model()

    def build_model(self) -> None:
        """
        Build the XMILE model string from the current model components.
        Writes the complete XMILE string to a file.
        """
        xmile_closing = """
    </xmile>
    """
        model_string = """
        <model>
            <variables>""" + f"{self.model}" + """
            </variables>
        </model>"""

        self.xmile_string = self.xmile_header + self.model_specs + model_string + xmile_closing

        with open('se-lib.xmile', 'w') as f:
            f.write(self.xmile_string)

    def add_stock(self, name: str, initial: float, inflows: List[str] = None, outflows: List[str] = None) -> None:
        """
        Add a stock to the model.

        Parameters
        ----------
        name : str
            The name of the stock
        initial : float
            Initial value of stock at start of simulation
        inflows : list of str, optional
            The names of the inflows to the stock
        outflows : list of str, optional
            The names of the outflows to the stock
        """
        if inflows is None:
            inflows = []
        if outflows is None:
            outflows = []

        inflow_string, outflow_string = "", ""
        for flow in inflows:
            inflow_string += f"""<inflow>"{flow}"</inflow>"""
        for flow in outflows:
            outflow_string += f"""<outflow>"{flow}"</outflow>"""

        self.model += f"""
                <stock name="{name}">
                    <doc>{name}</doc>
                    {inflow_string}
                    {outflow_string}
                    <eqn>{initial}</eqn>
                </stock>"""

        self.build_model()
        self.model_dict['stocks'][name] = {'inflows': inflows, 'outflows': outflows}

    def add_auxiliary(self, name: str, equation: str, inputs: List[str] = None) -> None:
        """
        Add auxiliary equation or constant to the model.

        Parameters
        ----------
        name : str
            The name of the auxiliary
        equation : str
            Equation for the auxiliary using other named model variables
        inputs : list, optional
            Optional list of variable input names used to draw model diagram
        """
        if inputs is None:
            inputs = []

        equation_str = str(equation)
        if "random()" in equation_str:
            equation_str = self._convert_random_to_xmile(equation_str)
        if "random.uniform(" in equation_str:
            equation_str = self._convert_random_to_xmile(equation_str)
        if "RANDOM" in equation_str:
            equation_str = equation_str.replace("RANDOM", '(GET_TIME_VALUE(0,0,0) + .00001) / (GET_TIME_VALUE(0,0,0) + .00001) * RANDOM')

        self.model += f"""
                <aux name="{name}">
                    <doc>{name}</doc>
                    <eqn>{equation_str}</eqn>
                </aux>"""

        self.build_model()
        self.model_dict['auxiliaries'][name] = {'equation': equation_str, 'inputs': inputs}

    def add_auxiliary_lookup(self, name: str, input_var: str, xpoints: List[float],
                            ypoints: List[float], inputs: List[str] = None) -> None:
        """
        Add auxiliary lookup table or graph to the model.

        Parameters
        ----------
        name : str
            The name of the auxiliary lookup
        input_var : str
            Input for the lookup function as the x-value which may be a constant, model variable or equation
        xpoints : list of float
            list of x points for the lookup function
        ypoints : list of float
            list of y points for the lookup function
        inputs : list, optional
            Optional list of variable input names used to draw model diagram
        """
        if inputs is None:
            inputs = []

        xpoint_string = ",".join(str(x) for x in xpoints)
        ypoint_string = ",".join(str(y) for y in ypoints)

        self.model += f"""
                <aux name="{name}">
                    <eqn>{input_var}</eqn>
                    <units>Fraction</units>
                    <doc>{name}</doc>
                    <gf>
                    <xpts>
                    {xpoint_string}
                    </xpts>
                    <ypts>
                    {ypoint_string}
                    </ypts>
                    </gf>
                </aux>"""

        self.build_model()
        self.model_dict['auxiliaries'][name] = {
            'input': input_var,
            'xpoints': xpoints,
            'ypoints': ypoints,
            'inputs': inputs
        }

    def add_flow(self, name: str, equation: str) -> None:
        """
        Add a flow to the model.

        Parameters
        ----------
        name : str
            The name of the flow
        equation : str
            Equation for the flow using other named model variables
        """
        equation_str = str(equation)
        if "random()" in equation_str:
            equation_str = self._convert_random_to_xmile(equation_str)
        if "random.uniform(" in equation_str:
            equation_str = self._convert_random_to_xmile(equation_str)
        if "RANDOM" in equation_str:
            equation_str = equation_str.replace("RANDOM", '(GET_TIME_VALUE(0,0,0) + .00001) / (GET_TIME_VALUE(0,0,0) + .00001) * RANDOM')

        self.model += f"""
                <flow name="{name}">
                    <doc>{name}</doc>
                    <eqn>{equation_str}</eqn>
                </flow>"""

        self.build_model()
        self.model_dict['flows'][name] = {'equation': equation_str, 'inputs': []}

    def _convert_random_to_xmile(self, equation: str) -> str:
        """
        Convert Python random functions to XMILE-compatible functions.

        Parameters
        ----------
        equation : str
            Equation string potentially containing random function calls

        Returns
        -------
        str
            Converted equation with XMILE-compatible random functions
        """
        equation = equation.replace("random(", "RANDOM_0_1(")
        equation = equation.replace("random.uniform(", "RANDOM_UNIFORM(")
        return equation

    def extract_flow_and_aux_dependencies(self) -> None:
        """
        Extract and update flow and auxiliary dependencies by analyzing the XMILE file.
        """
        xmile_path = 'se-lib.xmile'
        ns = {"ns": "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"}

        tree = ET.parse(xmile_path)
        root = tree.getroot()
        variables = root.findall(".//ns:model/ns:variables/*", ns)

        for var in variables:
            tag = var.tag.split("}")[-1]
            if tag not in ["flow", "aux"]:
                continue

            name = var.attrib.get("name")
            eqn_elem = var.find("ns:eqn", ns)

            if name and eqn_elem is not None:
                eqn_text = eqn_elem.text or ""
                inputs = [
                    var for var in set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", eqn_text)) - {name}
                    if var in self.model_dict.get('flows', {})
                    or var in self.model_dict.get('auxiliaries', {})
                    or var in self.model_dict.get('stocks', {})
                ]

                if tag == "aux":
                    if 'auxiliaries' in self.model_dict and name in self.model_dict['auxiliaries']:
                        self.model_dict['auxiliaries'][name]['inputs'] = inputs
                elif tag == "flow":
                    if 'flows' in self.model_dict and name in self.model_dict['flows']:
                        self.model_dict['flows'][name]['inputs'] = inputs

    def run_model(self, verbose: bool = True) -> pd.DataFrame:
        """
        Execute the current model.

        Parameters
        ----------
        verbose : bool, optional
            Whether to display verbose output, by default True

        Returns
        -------
        pd.DataFrame
            Pandas dataframe containing run outputs for each variable at each timestep
        """
        import pysd

        filename = './se-lib.xmile'
        model = pysd.read_xmile(filename)

        # Extract stop condition from XMILE header
        tree = ET.parse(filename)
        root = tree.getroot()
        ns = {'ns': 'http://docs.oasis-open.org/xmile/ns/XMILE/v1.0'}
        stop_when_elem = root.find('.//ns:stop_when', ns)
        stop_expression = stop_when_elem.text.strip() if stop_when_elem is not None else None

        if stop_expression:
            stop_expression = re.sub(r'\bAND\b', 'and', stop_expression, flags=re.IGNORECASE)
            stop_expression = re.sub(r'\bOR\b', 'or', stop_expression, flags=re.IGNORECASE)
            stop_expression = re.sub(r'\bNOT\b', 'not', stop_expression, flags=re.IGNORECASE)
            stop_expression = re.sub(
                r'\bIF\b\s+(.*?)\s+\bTHEN\b\s+(.*?)\s+\bELSE\b\s+(.*)',
                r'(\2) if (\1) else (\3)',
                stop_expression,
                flags=re.IGNORECASE
            )


        stop_condition_fn = None
        tracked_vars = []

        if stop_expression:
            # Parse the expression to extract variables
            vars_in_expr = [v for v in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", stop_expression)
                  if v in model.doc['Real Name'].values]
            tracked_vars = list(set(vars_in_expr))

            def stop_condition_fn(model):
                local_env = {var: model[var] for var in tracked_vars}
                return eval(stop_expression, {}, local_env)

        try:
            from pysd.py_backend.output import ModelOutput
        except ImportError:
            # Fall back to regular run if ModelOutput not available
            self.run_output = model.run(progress=False)
            return self.run_output

        # --- Optional Step-by-Step Execution ---
        if stop_condition_fn and tracked_vars:
            output_collector = ModelOutput()
            model.set_stepper(output_collector, step_vars=tracked_vars, final_time=model['FINAL TIME'])

            max_steps = int(model['FINAL TIME'] / model['TIME STEP'])
            for _ in range(max_steps):
                model.step(1)
                if stop_condition_fn(model):
                    break

            self.run_output = output_collector.collect(model)

            # Set true final time
            actual_final_time = self.run_output.index[-1]

            # Update the model's internal time
            model.components.time._time = actual_final_time

            if verbose:
                print(f"Simulation ends at time = {self.run_output.index[-1]}")

            # Try to update final_time if it exists as a component
            try:
                # First, try to directly access final_time if defined as a component
                if hasattr(model.components, 'final_time'):
                    final_time_func = model.components.final_time
                    if hasattr(final_time_func, '__call__'):
                        # Override the function's return value
                        final_time_func.__globals__['__return__'] = lambda: actual_final_time
                        # Update any cached value if present
                        if hasattr(final_time_func, 'metadata'):
                            final_time_func.metadata['cached_value'] = actual_final_time
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not override FINAL TIME — {e}")

            # Directly set _final_time if that attribute exists on the model
            if hasattr(model, '_final_time'):
                model._final_time = actual_final_time

        else:
            self.run_output = model.run(progress=False)

        try:
            if verbose:
                from IPython.display import display
                display(self.run_output)
        except Exception:
            pass

        return self.run_output

    def get_stop_time(self) -> float:
        """
        Returns the logical stop time in lieu of the fixed end time for a simulation.

        Returns
        -------
        float
            End time when the logical condition expires
        """
        if self.run_output is None:
            raise ValueError("Model has not been run yet. Call run_model() first.")
        return self.run_output.index[-1]

    def plot_graph(self, *run_outputs) -> None:
        """
        Display matplotlib graph for each model variable.

        Parameters
        ----------
        *run_outputs : str or list
            Variable name(s) or lists of variable names to plot on single graphs
        """
        if self.run_output is None:
            raise ValueError("Model has not been run yet. Call run_model() first.")

        for var in run_outputs:
            label_string = str(var)
            if type(var) == list:
                label_string = str(var[0])
                for count, element in enumerate(var):
                    if count > 0:
                        label_string += ", " + element

            fig, axis = plt.subplots(figsize=(6, 4))
            label_string = textwrap.fill(label_string, width=40)
            axis.set(xlabel='Time', ylabel=label_string)

            if type(var) == list:
                axis.plot(self.run_output.index, self.run_output[var].values, label=var)
                axis.legend(loc="best")
            else:
                axis.plot(self.run_output.index, self.run_output[var].values)

            plt.show()

    def save_graph(self, *run_outputs, filename="graph.png") -> None:
        """
        Save graph to file.

        Parameters
        ----------
        *run_outputs : str or list
            Variable name(s) or lists of variable names to plot on graph
        filename : str, optional
            File name with format extension, by default "graph.png"
        """
        if self.run_output is None:
            raise ValueError("Model has not been run yet. Call run_model() first.")

        for var in run_outputs:
            label_string = str(var)
            if type(var) == list:
                label_string = str(var[0])
                for count, element in enumerate(var):
                    if count > 0:
                        label_string += ", " + element

            fig, axis = plt.subplots(figsize=(6, 4))
            label_string = textwrap.fill(label_string, width=40)
            axis.set(xlabel='Time', ylabel=label_string)

            if type(var) == list:
                axis.plot(self.run_output.index, self.run_output[var].values, label=var)
                axis.legend(loc="best")
            else:
                axis.plot(self.run_output.index, self.run_output[var].values)

            plt.savefig(filename)

    def _is_inflow_of_stock(self, flow_name: str) -> bool:
        """
        Check if the given flow_name is an inflow of at least one stock in the model dictionary.

        Parameters
        ----------
        flow_name : str
            The name of the flow to check

        Returns
        -------
        bool
            True if flow_name is an inflow of at least one stock, False otherwise
        """
        for stock, stock_data in self.model_dict['stocks'].items():
            if flow_name in stock_data['inflows']:
                return True
        return False

    def _is_outflow_of_stock(self, flow_name: str) -> bool:
        """
        Check if the given flow_name is an outflow of at least one stock in the model dictionary.

        Parameters
        ----------
        flow_name : str
            The name of the flow to check

        Returns
        -------
        bool
            True if flow_name is an outflow of at least one stock, False otherwise
        """
        for stock, stock_data in self.model_dict['stocks'].items():
            if flow_name in stock_data['outflows']:
                return True
        return False

    def _render_graphviz(self, dot, format='svg'):
        """
        Render a Graphviz graph for inline display in IPython, specifically for system dynamics models
        to suppress label warnings on non-Windows platforms.

        Parameters
        ----------
        dot : graphviz.Digraph or graphviz.Graph
            The graph to render
        format : str, optional
            Output format ('svg', 'png', or 'jpg'), by default 'svg'

        Returns
        -------
        IPython display object or graphviz object
            The rendered image, or the original graph on Windows
        """
        def in_ipython():
            try:
                from IPython import get_ipython
                return get_ipython() is not None
            except ImportError:
                return False

        if platform.system() == 'Windows':
            return dot  # Skip rendering on Windows to avoid subprocess/file issues

        display_map = {
            'svg': SVG,
            'png': Image,
            'jpg': Image
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dot_path = os.path.join(tmpdir, 'input.dot')
            with open(dot_path, 'w', encoding='utf-8') as f:
                f.write(dot.source)

            cmd = ['dot', f'-T{format}', dot_path]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL  # Suppress Graphviz warnings
            )

            data = result.stdout
            if in_ipython():
                return display_map.get(format, print)(data)
            else:
                output_filename = os.path.abspath(f"graph_output.{format}")
                with open(output_filename, 'wb') as out_file:
                    out_file.write(data)
                print(f"Graph rendered and saved to: {output_filename}")

                # Try to open in system viewer
                if platform.system() == 'Darwin':
                    subprocess.run(['open', output_filename])
                elif platform.system() == 'Windows':
                    os.startfile(output_filename)
                elif platform.system() == 'Linux':
                    subprocess.run(['xdg-open', output_filename])

    def draw_model(self, filename=None, format='svg'):
        """
        Draw a diagram of the model.

        Parameters
        ----------
        filename : str, optional
            A filename for the output not including a filename extension
        format : str, optional
            The file format of the graphic output ('svg', 'png', etc.)

        Returns
        -------
        graphviz.Digraph
            Graphviz diagram of the system dynamics model
        """
        self.extract_flow_and_aux_dependencies()

        graph = graphviz.Digraph(engine='dot', filename=filename, format=format)
        graph.attr(rankdir='LR', size='10,8', splines='spline', margin="0.5,0")
        graph.attr('node', fontname="arial", fontcolor='blue', color='invis', fontsize='10')

        with graph.subgraph(name='cluster_flowchain') as c:
            # Add stocks as boxes
            for stock_name in self.model_dict['stocks']:
                graph.node(stock_name, shape='box', color='blue')

            # Add flows as circles
            for flow_name in self.model_dict['flows']:
                graph.node(flow_name, shape='circle', color='blue', width='.2', fixedsize="true", label=f"\n\n{flow_name}")

        # Add auxiliaries as circles
        for aux_name in self.model_dict['auxiliaries']:
            graph.node(aux_name, shape='circle', color='blue', width='.2', fixedsize="true", label=f"\n\n{aux_name}")

        # Add edges from inflows to stocks
        for stock_name, stock_dict in self.model_dict['stocks'].items():
            for inflow_name in stock_dict['inflows']:
                graph.edge(inflow_name, stock_name, color="blue:blue", arrowhead="onormal")
                if inflow_name not in stock_dict['outflows']:
                    # draw source if not an outflow of any other stock
                    if not self._is_outflow_of_stock(inflow_name):
                        graph.node(f'{inflow_name}_source', width=".01", fontsize='14', fixedsize="true", label="✽")
                        graph.edge(f'{inflow_name}_source', inflow_name, tailclip="true", color="blue:blue", arrowhead="none")

        # Add edges from stocks to outflows
        for stock_name, stock_dict in self.model_dict['stocks'].items():
            for outflow_name in stock_dict['outflows']:
                graph.edge(stock_name, outflow_name, color="blue:blue", arrowhead="none")
                if outflow_name not in stock_dict['inflows']:
                    # draw sink if not an inflow of any other stock
                    if not self._is_inflow_of_stock(outflow_name):
                        graph.node(f'{outflow_name}_sink', width=".01", fontsize='14', fixedsize="true", label="✽")
                        graph.edge(outflow_name, f'{outflow_name}_sink', headclip="true", color="blue:blue", arrowhead="onormal")

        # Add edges from variable inputs to flows
        for flow_name, flow_dict in self.model_dict['flows'].items():
            for input_name in flow_dict['inputs']:
                if input_name in self.model_dict['stocks']:
                    inflows = self.model_dict['stocks'][input_name].get('inflows', [])
                    outflows = self.model_dict['stocks'][input_name].get('outflows', [])
                    if flow_name in inflows:
                        tailport = 'nw'
                    elif flow_name in outflows:
                        tailport = 'ne'
                    else:
                        tailport = None
                    graph.edge(input_name, flow_name, color="red", arrowhead="normal", constraint='false',
                            tailport=tailport if tailport else None)
                else:
                    graph.edge(input_name, flow_name, color="red", arrowhead="normal", constraint='false')

        # Add edges from variable inputs to auxiliaries
        for aux_name, aux_dict in self.model_dict['auxiliaries'].items():
            for input_name in aux_dict['inputs']:
                graph.edge(input_name, aux_name, color="red", arrowhead="normal", constraint='false')

        if filename is not None:
            graph.render(quiet=True)

        # Conditionally display in notebooks
        try:
            from IPython.display import display
            display(self._render_graphviz(graph))
            return None
        except Exception:
            pass  # Not in an IPython notebook or display() unavailable

        return graph

    def get_model_structure(self) -> Dict:
        """
        Get the model structure.

        Returns
        -------
        Dict
            Model structure dictionary with stocks, flows, and auxiliaries
        """
        return self.model_dict


# Helper functions outside the class to maintain compatibility with the original API
def init_sd_model(start, stop, dt, stop_when=None):
    """
    Initialize a system dynamics model.

    Parameters
    ----------
    start : float
        Start time of the simulation
    stop : float
        Stop time of the simulation
    dt : float
        Time step
    stop_when : str, optional
        Optional logical condition to stop the simulation

    Returns
    -------
    SystemDynamicsModel
        A new system dynamics model instance
    """
    global sd_model
    sd_model = SystemDynamicsModel(start, stop, dt, stop_when)
    return sd_model

def add_stock(name, initial, inflows=None, outflows=None):
    """
    Add a stock to the current model.

    Parameters
    ----------
    name : str
        The name of the stock
    initial : float
        Initial value of stock at start of simulation
    inflows : list of str, optional
        The names of the inflows to the stock
    outflows : list of str, optional
        The names of the outflows to the stock
    """
    if inflows is None:
        inflows = []
    if outflows is None:
        outflows = []
    sd_model.add_stock(name, initial, inflows, outflows)

def add_auxiliary(name, equation, inputs=None):
    """
    Add auxiliary equation or constant to the current model.

    Parameters
    ----------
    name : str
        The name of the auxiliary
    equation : str
        Equation for the auxiliary using other named model variables
    inputs : list, optional
        Optional list of variable input names used to draw model diagram
    """
    if inputs is None:
        inputs = []
    sd_model.add_auxiliary(name, equation, inputs)

def add_auxiliary_lookup(name, input_var, xpoints, ypoints, inputs=None):
    """
    Add auxiliary lookup table or graph to the current model.

    Parameters
    ----------
    name : str
        The name of the auxiliary lookup
    input_var : str
        Input for the lookup function as the x-value
    xpoints : list of float
        list of x points for the lookup function
    ypoints : list of float
        list of y points for the lookup function
    inputs : list, optional
        Optional list of variable input names used to draw model diagram
    """
    if inputs is None:
        inputs = []
    sd_model.add_auxiliary_lookup(name, input_var, xpoints, ypoints, inputs)

def add_flow(name, equation):
    """
    Add a flow to the current model.

    Parameters
    ----------
    name : str
        The name of the flow
    equation : str
        Equation for the flow using other named model variables
    """
    sd_model.add_flow(name, equation)

def run_model(verbose=True):
    """
    Execute the current model.

    Parameters
    ----------
    verbose : bool, optional
        Whether to display verbose output, by default True

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing run outputs for each variable at each timestep
    """
    return sd_model.run_model(verbose)

def plot_graph(*run_outputs):
    """
    Display matplotlib graph for each model variable.

    Parameters
    ----------
    *run_outputs : str or list
        Variable name(s) or lists of variable names to plot on single graphs
    """
    sd_model.plot_graph(*run_outputs)

def save_graph(*run_outputs, filename="graph.png"):
    """
    Save graph to file.

    Parameters
    ----------
    *run_outputs : str or list
        Variable name(s) or lists of variable names to plot on graph
    filename : str, optional
        File name with format extension, by default "graph.png"
    """
    sd_model.save_graph(*run_outputs, filename=filename)

def draw_sd_model(filename=None, format='svg'):
    """
    Draw a diagram of the current model.

    Parameters
    ----------
    filename : str, optional
        A filename for the output not including a filename extension
    format : str, optional
        The file format of the graphic output ('svg', 'png', etc.)

    Returns
    -------
    graphviz.Digraph
        Graphviz diagram of the system dynamics model
    """
    return sd_model.draw_model(filename, format)

def get_stop_time():
    """
    Returns the logical stop time in lieu of the fixed end time for a simulation.

    Returns
    -------
    float
        End time when the logical condition expires
    """
    return sd_model.get_stop_time()

def get_model_structure():
    """
    Get the model structure.

    Returns
    -------
    Dict
        Model structure dictionary with stocks, flows, and auxiliaries
    """
    return sd_model.get_model_structure()

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

####################### DISCRETE EVENT ###############################
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

def plot_utilization(data, xlabel="Time", ylabel="Utilization", title="Server Utilization Over Time"):
    """
    Plot the server utilization over time as a step function.

    Parameters:
    -----------
    data : list of tuples
        Each tuple contains a timestamp and the corresponding utilization.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the graph.

    Returns:
    --------
    A Matplotlib plot.
    """
    # Extract time and utilization values from the data
    time_values, utilization_values = zip(*data)

    # Add the end time for the last utilization value
    time_values = list(time_values) + [time_values[-1] + 1]
    utilization_values = list(utilization_values) + [utilization_values[-1]]

    # Create the plot
    plt.figure(figsize=(10,6))
    plt.step(time_values, utilization_values, where='post', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.ylim(0, max(utilization_values) + 1)
    plt.show()