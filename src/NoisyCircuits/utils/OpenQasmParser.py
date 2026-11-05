# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
Helper module to parse OpenQASM 3.0 files and extract the quantum circuit instructions. 

Example:
    >>> from NoisyCircuits.utils import Parser
    >>> parser = Parser(file_path="path/to/circuit.qasm", instruction_list=[], append_to_circuit=False, basis_gates=[["x", "sx", "rz"], ["ecr"]])
    >>> circuit_instructions = parser.parse()
"""

import openqasm3
import numpy as np
from openqasm3 import ast, parser
import os


class NotAnEmptyCircuitError(Exception):
    """
    Error raised when the Quantum Circuit already contains instructions and append_circuit is set to False.
    """
    pass

_CONSTANTS = {
    "pi" : np.pi, "π" : np.pi,
    "tau" : 2 * np.pi, "τ" : 2 * np.pi,
    "euler" : np.e, "ℇ" : np.e
}

_FUNCTIONS = {
    "sin" : np.sin, "cos" : np.cos, "tan" : np.tan,
    "arcsin" : np.arcsin, "arccos" : np.arccos, "arctan" : np.arctan,
    "exp" : np.exp, "log" : np.log, "sqrt" : np.sqrt
}

_BINARY_OPERATIONS = {
    "+" : lambda a, b: a + b,
    "-" : lambda a, b: a - b,
    "*" : lambda a, b: a * b,
    "/" : lambda a, b: a / b,
    "**" : lambda a, b: a ** b
}

_IGNORED_STATEMENTS = (
    ast.Include,
    ast.ClassicalDeclaration,
    ast.QuantumGateDefinition,
    ast.QuantumBarrier,
    ast.DelayInstruction,
    ast.Pragma
)

class Parser:
    """
    Helper Class to parse OpenQASM 3.0 files to extract the quantum circuit and write it to the instruction list for use in the simulator.

    Supported are gate calls on single qubits of declared registers (``q[0]``, ``q[-1]``, scalar ``qubit r;``) or on physical qubits (``$3``), with at most one parameter given as a constant expression (numbers, ``pi``/``tau``/``euler``, ``+ - * / **``, unary ``-`` and the functions in ``_FUNCTIONS``). Include statements, classical declarations, gate definitions, barriers, delays and pragmas are ignored, and measurements are allowed only at the end of the circuit. Everything else raises a ValueError instead of being skipped.

    Parameters
    ----------
    file_path : str
        The path to the OpenQASM 3.0 file to be parsed.
    instruction_list : list
        The list to which the parsed instructions will be appended.
    append_circuit : bool
        A flag to indicate whether to add instructions to an existing quantum circuit or whether it needs to be a new circuit.
    basis_gates : list[list[str]]
        The basis gates of the quantum device.
    qubit_layout : list[int] | None
        Physical qubits of the circuit (logical qubit i is qubit_layout[i]). Physical qubits ``$p`` in the file are mapped to qubit_layout.index(p). If None, ``$p`` is mapped to logical qubit p.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    TypeError
        If append_circuit is not a boolean.
    NotAnEmptyCircuitError
        If the Quantum Circuit already contains instructions and append_circuit is set to False.
    ValueError
        If basis_gates is None.
    """
    def __init__(self,
                file_path : str,
                instruction_list : list,
                append_to_circuit : bool = False,
                basis_gates : list[list[str]] = None,
                qubit_layout : list[int] | None = None
                )->None:
        """
        Initializes the Parser object with the given file path, instruction list, and append circuit flag.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.file_path = file_path
        if not isinstance(append_to_circuit, bool):
            raise TypeError("append_to_circuit must be a boolean.")
        if instruction_list != [] and not append_to_circuit:
            raise NotAnEmptyCircuitError("The Quantum Circuit already contains instructions. Cannot add instructions when append_circuit is set to False")
        self.instruction_list = instruction_list
        if basis_gates is None:
            raise ValueError("basis_gates cannot be None. Please provide a list of supported gates.")
        self.basis_gates = basis_gates
        self.qubit_layout = qubit_layout

    def _check_instructions_against_supported_gates(self,
                                                    instruction_list:list[list[str, list[int], float|np.ndarray]]
                                                    )->None:
        """
        Helper private method to check if the parsed instructions are supported by the simulator based on the provided basis gates.

        Parameters
        ----------
        instruction_list : list[list[str, list[int], float|np.ndarray]]
            The list of parsed instructions to be checked against the supported gates.
        
        Raises
        ------
        ValueError
            If any of the parsed instructions contain gates that are not supported by the simulator.
        """
        supported_gates = [gate for sublist in self.basis_gates for gate in sublist]
        unsupported_gate_list = []
        supported_gates.append("unitary")
        for instruction in instruction_list:
            if instruction[0] not in supported_gates:
                unsupported_gate_list.append(instruction[0])
        if unsupported_gate_list != []:
            raise ValueError(f"The following gates are not supported by the simulator: {unsupported_gate_list}. Supported gates are: {supported_gates}")

    def _evaluate_parameter(self,
                            expression : ast.Expression,
                            line : int
                            )->float:
        """
        Helper private method to evaluate a constant gate parameter expression.

        Parameters
        ----------
        expression : ast.Expression
            The parameter expression of a gate call.
        line : int
            The line of the gate call in the OpenQASM file (for error messages).

        Returns
        -------
        float
            The value of the expression.

        Raises
        ------
        ValueError
            If the expression contains unknown identifiers, unsupported operations or does not evaluate to a finite number.
        """
        if isinstance(expression, (ast.IntegerLiteral, ast.FloatLiteral)):
            value = float(expression.value)
        elif isinstance(expression, ast.Identifier):
            if expression.name not in _CONSTANTS:
                raise ValueError(f"Line {line}: unknown identifier '{expression.name}' in gate parameter. Only numbers and the constants {list(_CONSTANTS)} are supported.")
            value = _CONSTANTS[expression.name]
        elif isinstance(expression, ast.UnaryExpression) and expression.op.name in ("-", "+"):
            value = self._evaluate_parameter(expression.expression, line)
            if expression.op.name == "-":
                value = -value
        elif isinstance(expression, ast.BinaryExpression) and expression.op.name in _BINARY_OPERATIONS:
            lhs = self._evaluate_parameter(expression.lhs, line)
            rhs = self._evaluate_parameter(expression.rhs, line)
            try:
                value = _BINARY_OPERATIONS[expression.op.name](lhs, rhs)
            except (ZeroDivisionError, OverflowError) as error:
                raise ValueError(f"Line {line}: gate parameter cannot be evaluated ({error}).") from None
        elif isinstance(expression, ast.FunctionCall) and expression.name.name in _FUNCTIONS and len(expression.arguments) == 1:
            with np.errstate(all="ignore"):
                value = float(_FUNCTIONS[expression.name.name](self._evaluate_parameter(expression.arguments[0], line)))
        else:
            raise ValueError(f"Line {line}: unsupported gate parameter expression of type {type(expression).__name__}.")
        if not isinstance(value, float) or not np.isfinite(value):
            raise ValueError(f"Line {line}: gate parameter does not evaluate to a finite real number.")
        return value

    def _declare_register(self,
                          statement : ast.QubitDeclaration
                          )->None:
        """
        Helper private method to register a qubit declaration. Registers are placed one after the other in the order of declaration.

        Parameters
        ----------
        statement : ast.QubitDeclaration
            The qubit declaration statement.

        Raises
        ------
        ValueError
            If the register is declared twice, its size is not a positive integer literal or physical qubits are used in the same file.
        """
        line = statement.span.start_line
        name = statement.qubit.name
        if self._uses_physical_qubits:
            raise ValueError(f"Line {line}: qubit declarations cannot be mixed with physical qubits ($n).")
        if name in self._registers:
            raise ValueError(f"Line {line}: qubit register '{name}' is declared more than once.")
        if statement.size is None:
            size = None
        elif isinstance(statement.size, ast.IntegerLiteral) and statement.size.value > 0:
            size = statement.size.value
        else:
            raise ValueError(f"Line {line}: size of qubit register '{name}' must be a positive integer literal.")
        self._registers[name] = (self._num_declared_qubits, size)
        self._num_declared_qubits += 1 if size is None else size

    def _resolve_qubit(self,
                       qubit : ast.Identifier | ast.IndexedIdentifier,
                       line : int
                       )->int:
        """
        Helper private method to convert a gate operand into the logical qubit index used by the simulator.

        Parameters
        ----------
        qubit : ast.Identifier | ast.IndexedIdentifier
            The gate operand.
        line : int
            The line of the gate call in the OpenQASM file (for error messages).

        Returns
        -------
        int
            The logical qubit index.

        Raises
        ------
        ValueError
            If the operand is an undeclared register, a whole register, a slice or set, an index out of range, or a physical qubit that is not in the qubit layout.
        """
        if isinstance(qubit, ast.Identifier) and qubit.name.startswith("$"):
            if self._registers:
                raise ValueError(f"Line {line}: physical qubits ($n) cannot be mixed with qubit declarations.")
            self._uses_physical_qubits = True
            physical = int(qubit.name[1:])
            if self.qubit_layout is None:
                return physical
            if physical not in self.qubit_layout:
                raise ValueError(f"Line {line}: physical qubit {qubit.name} is not part of the circuit's qubit layout {self.qubit_layout}.")
            return self.qubit_layout.index(physical)
        name = qubit.name if isinstance(qubit, ast.Identifier) else qubit.name.name
        if name not in self._registers:
            raise ValueError(f"Line {line}: qubit register '{name}' is not declared.")
        offset, size = self._registers[name]
        if isinstance(qubit, ast.Identifier):
            if size is None:
                return offset
            raise ValueError(f"Line {line}: gates on whole registers ('{name}') are not supported. Apply the gate to each qubit '{name}[i]' instead.")
        if size is None:
            raise ValueError(f"Line {line}: '{name}' is a single qubit and cannot be indexed.")
        if len(qubit.indices) != 1 or not isinstance(qubit.indices[0], list) or len(qubit.indices[0]) != 1 or isinstance(qubit.indices[0][0], ast.RangeDefinition):
            raise ValueError(f"Line {line}: slices, sets and multiple indices of '{name}' are not supported. Use one integer index per qubit.")
        index = qubit.indices[0][0]
        sign = 1
        if isinstance(index, ast.UnaryExpression) and index.op.name == "-":
            index, sign = index.expression, -1
        if not isinstance(index, ast.IntegerLiteral):
            raise ValueError(f"Line {line}: index of '{name}' must be an integer literal.")
        position = sign * index.value
        if position < -size or position >= size:
            raise ValueError(f"Line {line}: index {position} is out of range for qubit register '{name}' of size {size}.")
        return offset + position % size

    def parse(self)->list[list[str, list[int], float|np.ndarray]]:
        """
        Method to parse the OpenQASM file.

        Returns
        -------
        list[list[str, list[int], float|np.ndarray]]
            A list of instructions with each instruction represented by a list of gate, qubits and parameters. Generated to be compatible with the NoisyCircuits simulators.

        Raises
        ------
        ValueError
            If the file contains statements, gate modifiers, operands or parameters that cannot be simulated.
        """
        with open(self.file_path, "r") as f:
            qasm_code = f.read()
        program_circuit = parser.parse(qasm_code)
        self._registers = {}
        self._num_declared_qubits = 0
        self._uses_physical_qubits = False
        measured = False
        for statement in program_circuit.statements:
            line = statement.span.start_line if statement.span is not None else "?"
            if isinstance(statement, ast.QubitDeclaration):
                self._declare_register(statement)
                continue
            if isinstance(statement, ast.QuantumMeasurementStatement):
                measured = True
                continue
            if isinstance(statement, _IGNORED_STATEMENTS):
                continue
            if not isinstance(statement, ast.QuantumGate):
                raise ValueError(f"Line {line}: {type(statement).__name__} statements are not supported. Only gate calls, declarations, barriers, delays and final measurements can be imported.")
            gate_name = statement.name.name
            if measured:
                raise ValueError(f"Line {line}: gate '{gate_name}' after a measurement. Mid-circuit measurements are not supported; measurements must come at the end of the circuit.")
            if statement.modifiers:
                raise ValueError(f"Line {line}: gate modifiers ({', '.join(modifier.modifier.name for modifier in statement.modifiers)}) on gate '{gate_name}' are not supported.")
            if len(statement.arguments) > 1:
                raise ValueError(f"Line {line}: gate '{gate_name}' has {len(statement.arguments)} parameters, but at most one is supported.")
            qubit_list = [self._resolve_qubit(qubit, line) for qubit in statement.qubits]
            if len(set(qubit_list)) != len(qubit_list):
                raise ValueError(f"Line {line}: gate '{gate_name}' acts on the same qubit more than once.")
            if len(qubit_list) == 1:
                qubit_list.append(qubit_list[0])
            if statement.arguments != []:
                param = self._evaluate_parameter(statement.arguments[0], line)
            else:
                param = 0.0
            self.instruction_list.append([gate_name, qubit_list, param])
        self._check_instructions_against_supported_gates(self.instruction_list)    
        return self.instruction_list