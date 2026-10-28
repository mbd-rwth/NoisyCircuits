# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module provides the exact decomposition for several single and two qubit gates using only the gates that are available in the Heron QPU from IBM quantum hardware. The decomposition also considers the swap sequencing required for connecting qubits in the systems while keeping the directionality of qubits in mind. It is implemented with the abtract class `Decomposition` in the background to conform to error handling and skeleton templates. It is selected according to the user specified QPU in the QuantumCircuit class.
"""

import numpy as np
from NoisyCircuits.utils.SwapSequence import QubitCouplingMap
from NoisyCircuits.utils.Decomposition import Decomposition

class HeronDecomposition(Decomposition):
    """
    Class for Quantum Gate Application for the IBM Heron QPU Architectures
    """
    def __init__(self,
                 num_qubits:int,
                 connectivity:dict,
                 qubit_map:list[tuple],
                 use_fractional:bool = False
                 ):
        """
        Constructor for the HeronDecomposition class that applies the quantum gates for the IBM Heron QPU Architectures.

        Parameters
        ----------
        num_qubits : int
            The number of qubits in the quantum circuit.
        connectivity : dict
            A dictionary representing the connectivity of the qubits.
        qubit_map : list[tuple]
            A list of tuples representing the mapping of logical qubits to physical qubits.
        use_fractional : bool
            A flag to indicate whether to use fractional gates or not.
        """
        super().__init__(num_qubits=num_qubits)
        self.instruction_list = []
        self.num_qubits = num_qubits
        self.connectivity = connectivity
        self.qubit_map = qubit_map
        self.qubit_coupling = QubitCouplingMap(num_qubits=self.num_qubits, connectivity=self.connectivity)
        self.use_fractional = use_fractional

    def X(self,
          qubit:int):
        if super().X(qubit=qubit):
            self.instruction_list.append(["x", [qubit, qubit], 0])

    def SX(self,
           qubit:int):
        if super().SX(qubit=qubit):
            self.instruction_list.append(["sx", [qubit, qubit], 0])
    
    def RZ(self,
           theta:int|float,
           qubit:int):
        if super().RZ(theta=theta, qubit=qubit):
            self.instruction_list.append(["rz", [qubit, qubit], theta])
    
    def RX(self,
           theta:int|float,
           qubit:int):
        if super().RX(theta=theta, qubit=qubit):
            if self.use_fractional:
                self.instruction_list.append(["rx", [qubit, qubit], theta])
            else:
                self.RZ(theta=np.pi/2, qubit=qubit)
                self.SX(qubit=qubit)
                self.RZ(theta=2*np.pi + theta, qubit=qubit)
                self.SX(qubit=qubit)
                self.RZ(theta=5*np.pi/2, qubit=qubit)
                self.X(qubit=qubit)
    
    def RY(self,
           theta:int|float,
           qubit:int):
        if super().RY(theta=theta, qubit=qubit):
            self.X(qubit=qubit)
            self.SX(qubit=qubit)
            self.RZ(theta=-theta, qubit=qubit)
            self.SX(qubit=qubit)

    def H(self,
          qubit:int):
        if super().H(qubit=qubit):
            self.SX(qubit=qubit)
            self.RZ(theta=np.pi/2, qubit=qubit)
            self.SX(qubit=qubit)
    
    def Z(self,
          qubit:int):
        if super().Z(qubit=qubit):
            self.X(qubit=qubit)
            self.RX(theta=np.pi, qubit=qubit)
            self.RZ(theta=-np.pi, qubit=qubit)

    def Y(self,
          qubit:int):
        if super().Y(qubit=qubit):
            self.RY(theta=-np.pi, qubit=qubit)
            self.RX(theta=np.pi, qubit=qubit)
            self.X(qubit=qubit)
    
    def apply_swap_decomposition(self,
                                 qubit1:int,
                                 qubit2:int):
        if super().apply_swap_decomposition(qubit1=qubit1, qubit2=qubit2):
            self.SX(qubit=qubit1)
            self.SX(qubit=qubit2)
            self.CZ(control=qubit1, target=qubit2)
            self.SX(qubit=qubit1)
            self.SX(qubit=qubit2)
            self.CZ(control=qubit1, target=qubit2)
            self.SX(qubit=qubit1)
            self.SX(qubit=qubit2)
            self.CZ(control=qubit1, target=qubit2)
            self.RX(theta=np.pi, qubit=qubit1)
            self.X(qubit=qubit1)

    def CZ(self,
           control:int,
           target:int):
        if super().CZ(control=control, target=target):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            match_qubits = next((t for t in self.qubit_map if phys_control in t and phys_target in t), None)
            if phys_control == match_qubits[0] and phys_target == match_qubits[1]:
                self.instruction_list.append(["cz", [phys_control, phys_target], 0])
            else:
                self.instruction_list.append(["cz", [phys_target, phys_control], 0])
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def RZZ(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        if super().RZZ(theta=theta, qubit1=qubit1, qubit2=qubit2):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=qubit1, logical_target=qubit2)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            if self.use_fractional:
                self.instruction_list.append(["rzz", [phys_control, phys_target], theta])
            else:
               self.RZ(np.pi/2, phys_target)
               self.SX(phys_target)
               self.RZ(np.pi, phys_target)
               self.CZ(phys_control, phys_target)
               self.SX(phys_target)
               self.RZ(theta-np.pi, phys_target)
               self.SX(phys_target)
               self.RZ(np.pi, phys_target)
               self.CZ(phys_control, phys_target)
               self.SX(phys_target)
               self.RZ(np.pi/2, phys_target)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def RXX(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        if super().RXX(theta=theta, qubit1=qubit1, qubit2=qubit2):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=qubit1, logical_target=qubit2)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RZ(theta=-np.pi/2, qubit=phys_control)
            self.RX(theta=-np.pi/2, qubit=phys_control)
            self.RZ(theta=-np.pi/2, qubit=phys_control)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            self.RX(theta=-np.pi/2, qubit=phys_target)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            self.RZZ(theta=theta, qubit1=phys_control, qubit2=phys_target)
            self.RZ(theta=-np.pi/2, qubit=phys_control)
            self.RX(theta=-np.pi/2, qubit=phys_control)
            self.RZ(theta=-np.pi/2, qubit=phys_control)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            self.RX(theta=-np.pi/2, qubit=phys_target)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def RYY(self, 
            theta:int|float, 
            qubit1:int, 
            qubit2:int):
        if super().RYY(theta=theta, qubit1=qubit1, qubit2=qubit2):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=qubit1, logical_target=qubit2)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RX(theta=np.pi/2, qubit=phys_control)
            self.RX(theta=np.pi/2, qubit=phys_target)
            self.RZZ(theta=theta, qubit1=phys_control, qubit2=phys_target)
            self.RX(theta=-np.pi/2, qubit=phys_control)
            self.RX(theta=-np.pi/2, qubit=phys_target)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def CX(self,
           control:int,
           target:int):
        if super().CX(control=control, target=target):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.CZ(control=phys_control, target=phys_target)
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.RX(theta=-np.pi, qubit=phys_control)
            self.X(qubit=phys_control)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def ECR(self,
            control:int,
            target:int):
        if super().ECR(control=control, target=target):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RZ(theta=np.pi/2, qubit=phys_control)
            self.SX(qubit=phys_target)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            self.RX(theta=-np.pi/2, qubit=phys_target)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            self.CZ(control=phys_control, target=phys_target)
            self.X(qubit=phys_control)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            self.RX(theta=-np.pi/2, qubit=phys_target)
            self.RZ(theta=-np.pi/2, qubit=phys_target)
            self.RZ(theta=2*np.pi, qubit=phys_control)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def CY(self,
           control:int,
           target:int):
        if super().CY(control=control, target=target):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RX(theta=np.pi/2, qubit=phys_target)
            self.CZ(control=phys_control, target=phys_target)
            self.RX(theta=-np.pi/2, qubit=phys_target)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def CRX(self,
            theta:int|float,
            control:int,
            target:int):
        if super().CRX(theta=theta, control=control, target=target):    
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=(np.pi + theta)/2, qubit=phys_target)
            self.RZZ(theta=-theta/2, qubit1=phys_control, qubit2=phys_target)
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.RX(theta=-np.pi, qubit=phys_control)
            self.X(qubit=phys_control)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def CRY(self,
            theta:int|float,
            control:int,
            target:int):
        if super().CRY(theta=theta, control=control, target=target):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=3*np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=(5*np.pi + theta)/2, qubit=phys_target)
            self.RZZ(theta=-theta/2, qubit1=phys_control, qubit2=phys_target)
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=np.pi/2, qubit=phys_target)
            self.SX(qubit=phys_target)
            self.RZ(theta=5*np.pi/2, qubit=phys_target)
            self.RX(theta=2*np.pi, qubit=phys_control)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])

    def CRZ(self,
            theta:int|float,
            control:int,
            target:int):
        if super().CRZ(theta=theta, control=control, target=target):
            forward_swaps, reverse_swaps, phys_control, phys_target = self.qubit_coupling.generate_swap_sequence(logical_control=control, logical_target=target)
            for swap in forward_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
            self.RZ(theta=theta/2, qubit=phys_target)
            self.RZZ(theta=-theta/2, qubit1=phys_control, qubit2=phys_target)
            for swap in reverse_swaps:
                self.apply_swap_decomposition(qubit1=swap[0], qubit2=swap[1])
    
    def SWAP(self,
             qubit1:int,
             qubit2:int):
        if super().SWAP(qubit1=qubit1, qubit2=qubit2):
            physical1 = self.qubit_coupling.logical_to_physical[qubit1]
            physical2 = self.qubit_coupling.logical_to_physical[qubit2]
            if physical2 in self.qubit_coupling.connectivity.get(physical1, []) or physical1 in self.qubit_coupling.connectivity.get(physical2, []):
                self.apply_swap_decomposition(qubit1=physical1, qubit2=physical2)
            else:
                path = self.qubit_coupling.find_shortest_path(physical1, physical2)
                current_pos = physical1
                for i in range(1, len(path)):
                    next_pos = path[i]
                    self.apply_swap_decomposition(qubit1=current_pos, qubit2=next_pos)
                    current_pos = next_pos
                for i in range(len(path) - 2, 0, -1):
                    prev_pos = path[i-1]
                    curr_pos = path[i]
                    self.apply_swap_decomposition(qubit1=curr_pos, qubit2=prev_pos)
    
    def apply_unitary(self, 
                      unitary_matrix:np.ndarray,
                      qubits:list[int]):
        if super().apply_unitary(unitary_matrix, qubits):
            self.instruction_list.append(["unitary", qubits, unitary_matrix])