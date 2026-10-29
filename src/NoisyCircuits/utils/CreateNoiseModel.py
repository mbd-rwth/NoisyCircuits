# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module allows users to create a noise model from user provided calibration data in a CSV file format. The calibration data should include specific columns for qubit properties and gate errors. The class reads the calibration data, processes it, and constructs a noise model that can be used for quantum circuit simulations.

Alternatively, users can obtain the noise model from the calibration data of IBM Quantum Hardware using IBM Rest API. This class retrieves the calibration data of the specified backend and processes it to create the noise model in dictionary format.

See the documentation of the specific classes for more details on the expected format of the calibration data and the usage of the classes to create or obtain the noise model.
"""

from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError
from qiskit.quantum_info import average_gate_fidelity
import pandas as pd
import numpy as np
import requests
import os
import time

class CreateNoiseModel:
    """
    A class to create a noise model from user provided calibration data. The calibration data should be provided in a CSV file format with specific columns for qubit properties and gate errors. The class reads the calibration data, processes it, and constructs a noise model that can be used for quantum circuit simulations.

    Additionally, the user must specify the basis gates for the quantum hardware in order to create the noise model. The basis gates must be provided in a list of lists format where the first list contains the single qubit basis gates and the second list contains the two qubit basis gates. For example, if the single qubit basis gates are "x", "sx" and "rz" and the two qubit basis gate is "cz", then the basis gates should be provided as [["x", "sx", "rz"], ["cz"]].

    The definition of the header in the CSV file (in table view) should be as follows:

    +-------+---------+---------+-------------+-------------+--------------+--------------+-------------+--------------+
    | Qubit | T1 (us) | T2 (us) | Prob meas 0 | Prob meas 1 | Single Qubit | Single Qubit | Gate Length | Two Qubit    |
    |       |         |         | prep 1      | prep 0      | Gate Length  | Basis Gate   | (ns)        | Basis Gate   |
    |       |         |         |             |             | (ns)         | Error        |             | Error        |
    +=======+=========+=========+=============+=============+==============+==============+=============+==============+
    |       |         |         |             |             |              |              |             |              |
    +-------+---------+---------+-------------+-------------+--------------+--------------+-------------+--------------+

    An example of the content of the CSV file (in table view) where the basis gates are [["x", "sx"], ["cz"]] is as follows:

    +-------+---------+---------+--------------------+--------------------+-------------------------------+----------+----------+------------------+--------------------+
    | Qubit | T1 (us) | T2 (us) | Prob meas 0 prep 1 | Prob meas 1 prep 0 | Single Qubit Gate Length (ns) |  x Error | sx Error | Gate Length (ns) |      cz Error      |
    +=======+=========+=========+====================+====================+===============================+==========+==========+==================+====================+
    |   0   |  50.534 |  19.955 |       0.0789       |       0.1316       |               24              | 0.002118 | 0.002118 |       1:68       |      1:0.0207      |
    +-------+---------+---------+--------------------+--------------------+-------------------------------+----------+----------+------------------+--------------------+
    |   1   | 153.049 | 144.814 |       0.0696       |       0.1243       |               24              |  0.00052 |  0.00052 |     2:68;0:68    | 2:0.00479;0:0.0207 |
    +-------+---------+---------+--------------------+--------------------+-------------------------------+----------+----------+------------------+--------------------+

    For a better example of the content of the CSV file, please refer to the example CSV file provided in the NoisyCircuits repository within the noise_models directory.

    Parameters
    ----------
    calibration_data_file : str
        The path to the CSV file containing the calibration data.
    
    Raises
    ------
    TypeError:
        - calibration_data_file should be a string representing the path to the CSV file.
        - basis_gates should be a list of lists of strings representing the basis gates for the quantum hardware.
    FileNotFoundError
        If the specified CSV file is not found.
    ValueError
        If the CSV file does not contain the required columns or if the data is not in the expected format.

    Example
    -------
        >>> from NoisyCircuits.utils.CreateNoiseModel import CreateNoiseModel
        >>> calibration_file_path = "path/to/calibration_data.csv"
        >>> basis_gates = [["x", "sx", "rz"], ["cz"]]
        >>> noise_model = CreateNoiseModel(calibration_data_file=calibration_file_path, basis_gates=basis_gates).create_noise_model()
    """
    def __init__(self, 
                 calibration_data_file:str,
                 basis_gates:list[list[str]]
                )->None:
        """
        Constructor for the CreateNoiseModel class.
        """
        if not isinstance(calibration_data_file, str):
            raise TypeError("calibration_data_file must be a Path-like string")
        self.calibration_data_file = calibration_data_file
        if not isinstance(basis_gates, list) or not all(isinstance(gate_list, list) for gate_list in basis_gates) or not all(isinstance(gate, str) for gate_list in basis_gates for gate in gate_list):
            raise TypeError("basis_gates must be a list of lists of strings representing the basis gates for the quantum hardware")
        try:
            self.calibration_data = pd.read_csv(calibration_data_file, delimiter=",")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{calibration_data_file}' not found. Please check the file path and try again.")
        self.basis_gates = basis_gates
        column_names = self.calibration_data.columns.tolist()
        for gate_list in self.basis_gates:
            for gate in gate_list:
                if f"{gate.lower()} Error" not in column_names:
                    raise ValueError(f"CSV file is missing required column '{gate} Error' for gate '{gate}' in basis gates.")
        other_required_columns = ["Qubit", "T1 (us)", "T2 (us)", "Prob meas 0 prep 1", "Prob meas 1 prep 0", "Single Qubit Gate Length (ns)", "Gate Length (ns)"]
        for col in other_required_columns:
            if col not in column_names:
                raise ValueError(f"CSV file is missing required column '{col}'.")

    def _compute_depolarizing_probability(self, 
                                          gate_error:float,
                                          F_thermal:float,
                                          dim:int
                                          )->float:
        """
        Private method to compute the depolarizing probability.

        Parameters
        ----------
        gate_error : float
            The error rate of the gate.
        F_thermal : float
            The average gate fidelity of the thermal relaxation error.
        dim : int
            The dimension of the gate matrix

        Returns
        -------
        float
            The depolarizing probability.
        """
        depol_prob = dim * (gate_error - (1 - F_thermal)) / (dim * F_thermal - 1)
        n = int(np.log2(dim))
        max_depol_prob = 4**n / (4**n - 1)
        return min(max(0, depol_prob), max_depol_prob)

    def _parse_coupler_values(
            self,
            entry : str
        ) -> dict:
        """
        Method to parse the entry of a qubit coupler from the calibration data

        Parameters
        ----------
        entry : str
            Qubit coupler entry of a row in the calibration data with empty items read as NaN.

        Returns
        -------
        dict
            Mapping from the target qubit to the value, or empty for no data
        """
        if not isinstance(entry, str) or entry.strip() == "":
            return {}
        values = {}
        for item in entry.split(";"):
            target_qubit, value = item.split(":")
            values[int(target_qubit)] = float(value)
        return values
    
    def create_noise_model(self)->dict:
        """
        Method to create the noise model from the calibration data.

        Returns
        -------
        dict
            The noise model of the specified backend in dictionary format.
        """
        noise_model = NoiseModel()
        qubits = self.calibration_data["Qubit"].tolist()
        t1 = self.calibration_data["T1 (us)"].tolist()
        t2 = self.calibration_data["T2 (us)"].tolist()
        relaxation_time_anomaly_index = np.where(np.array(t2) > 2 * np.array(t1))[0]
        if len(relaxation_time_anomaly_index) > 0:
            for index in relaxation_time_anomaly_index:
                print(r"Warning: Found relaxation time anomaly for qubit {} with $T_2 \geq 2T_1$. Setting $T_2 = 2T_1$.".format(qubits[index]))
                t2[index] = 2 * t1[index]
        mean_errors = {}
        for gate in self.basis_gates[0]:
            mean_errors[gate] = self.calibration_data["{} Error".format(gate)].mean()
        for q in range(len(qubits)):
            qubits[q] = int(qubits[q])
            t1[q] = t1[q] * 1e-6
            t2[q] = t2[q] * 1e-6
        for i, row in self.calibration_data.iterrows():
            p01 = float(row["Prob meas 0 prep 1"])
            p10 = float(row["Prob meas 1 prep 0"])
            single_qubit_gate_length = int(row["Single Qubit Gate Length (ns)"]) * 1e-9
            two_qubit_gate_length = self._parse_coupler_values(row["Gate Length (ns)"])
            readout_matrix = np.array([[1 - p10, p10], [p01, 1 - p01]])
            noise_model.add_readout_error(ReadoutError(readout_matrix), [qubits[i]])
            for gate in self.basis_gates[0]:
                gate_error = row["{} Error".format(gate)]
                if np.isnan(gate_error):
                    gate_error = 1.0
                thermal_error = thermal_relaxation_error(t1[i], t2[i], single_qubit_gate_length)
                F_thermal = average_gate_fidelity(thermal_error)
                depol_prob = self._compute_depolarizing_probability(gate_error, F_thermal, 2)
                depol_error = depolarizing_error(depol_prob, 1)
                noise_model.add_quantum_error(depol_error.compose(thermal_error), gate, [qubits[i]])

            qubit_coupler_errors = {
                gate : self._parse_coupler_values(
                    row["{} Error".format(gate)]
                ) for gate in self.basis_gates[1]
            }
            for gate in self.basis_gates[1]:
                alt_error = [
                    qubit_coupler_errors[g] for g in self.basis_gates[1] if g != gate and qubit_coupler_errors[g]
                ]
                for target_qubit, gate_length in two_qubit_gate_length.items():
                    if target_qubit in qubit_coupler_errors[gate]:
                        gate_error = qubit_coupler_errors[gate][target_qubit]
                    elif alt_error and target_qubit in alt_error[0]:
                        gate_error = alt_error[0][target_qubit]
                        print(
                            "No error data for gate {} on qubits ({}, {}). Using error from other qubit couplers for the qubit pair.".format(gate, qubits[i], target_qubit)
                        )
                    else:
                        gate_error = 1.0
                        print(
                            "No error data found for any gate on the qubit coupler for pair ({}, {}), using maximum error.".format(qubits[i], target_qubit)
                        )
                    gate_time = gate_length * 1e-9
                    thermal_error = thermal_relaxation_error(
                        t1[target_qubit],
                        t2[target_qubit],
                        gate_time
                    ).tensor(
                        thermal_relaxation_error(
                            t1[i],
                            t2[i],
                            gate_time
                        )
                    )
                    depol_prob = self._compute_depolarizing_probability(
                        gate_error,
                        average_gate_fidelity(thermal_error),
                        4
                    )
                    depol_error = depolarizing_error(depol_prob, 2)
                    noise_model.add_quantum_error(
                        depol_error.compose(thermal_error),
                        gate,
                        [qubits[i], qubits[target_qubit]]
                    )
        return noise_model.to_dict(serializable=True)


class GetNoiseModel:
    """
    This class allows the user to obtain the noise model from the calibration data of IBM Quantum Hardware using IBM Rest API. This class retreives the calibration data of the specified backend and processes it to create the noise model in dictionary format. 

    A valid IBM Quantum API token and CRN is required to access the calibration data of the backend.

    Parameters
    ----------
    backend_name : str
        The name of the IBM Quantum backend.
    token : str
        The IBM Quantum API token.
    service_crn : str
        The CRN of the IBM Quantum service instance.

    Raises
    ------
    TypeError
        If backend_name, token or service_crn is not a string.
    ValueError
        If there is an issue connecting to the IBM Quantum API or if the backend is not found in the user's account.

    Example
    -------
        >>> from NoisyCircuits.utils.CreateNoiseModel import GetNoiseModel
        >>> backend_name = "ibm_perth"
        >>> token = "your_ibm_quantum_api_token"
        >>> service_crn = "your_ibm_quantum_service_crn"
        >>> noise_model_generator = GetNoiseModel(backend_name=backend_name, token=token, service_crn=service_crn)
        >>> noise_model = noise_model_generator.get_noise_model(save_csv=True, destination="path/to/save/csv", file_name="calibration_data.csv")
    """
    def __init__(self,
                 backend_name:str,
                 token:str,
                 service_crn:str
                )->None:
        """
        Constructor for the GetNoiseModel class.
        """
        if not isinstance(backend_name, str):
            raise TypeError("backend_name must be a string")
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if not isinstance(service_crn, str):
            raise TypeError("service_crn must be a string")
        self.backend_name = backend_name
        self.token = token
        self.service_crn = service_crn
        self._api_url = "https://quantum.cloud.ibm.com/api/v1/backends/{}/properties".format(backend_name)
        self._get_IAM_token()
        self._headers = {
            "Accept": "application/json",
            "IBM-API-Version": "2026-02-15",
            "Authorization": "Bearer {}".format(self._access_token),
            "Service-CRN": self.service_crn
        }
        self._basis_gates = self._get_basis_gates()

    def _get_basis_gates(self)->list[list[str]]:
        """
        Private method to obtain the basis gates for the specified backend from IBM Quantum REST API using the QPU family information.

        Returns
        -------
        list[list[str]]
            A list of lists containing the single qubit and two qubit basis gates for the specified backend.
        
        Raises
        ------
        ValueError
            If the API request fails or if the QPU family is not supported, an error is raised with the detailed error message from the API response or with the unsupported QPU family information.
        """
        basis_gates_url = "https://quantum.cloud.ibm.com/api/v1/backends/{}/configuration".format(self.backend_name)
        response = requests.request(
            "GET",
            basis_gates_url,
            headers = self._headers,
        )
        if response.status_code != 200:
            raise ValueError("Failed to retrieve basis gates from IBM Quantum API. Detailed error: {}".format(response.text))
        qpu_family = response.json()["processor_type"]["family"].lower()
        if qpu_family == "heron":
            basis_gates = [["x", "sx", "rz", "rx"], ["cz", "rzz"]]
        elif qpu_family == "eagle":
            basis_gates = [["x", "sx", "rz"], ["ecr"]]
        else:
            raise ValueError("Unsupported QPU Family: {}".format(qpu_family))
        return basis_gates

    def _get_IAM_token(self)->None:
        """
        Private method to obtain the IAM token for authentication with the IBM Quantum API using the provided token.

        Raises
        ------
        ValueError
            If the API request fails, an error is raised with the detailed error message from the API response.
        """
        iam_response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            data = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={}".format(self.token),
        )
        if iam_response.status_code != 200:
            raise ValueError("Failed to obtain IAM token from IBM Quantum API. Detailed error: {}".format(iam_response.text))
        self._access_token = iam_response.json()["access_token"]

    def _get_calibration_json(self)->None:
        """
        Private method of the class to obtain the raw calibration data in JSON format from the IBM Quantum API for the specified backend.

        Raises
        ------
        ValueError
            If the API request fails or if the backend is not found in the user's account, an error is raised with the detailed error message from the API response.
        """
        response = requests.request(
            "GET", 
            self._api_url,
            headers = self._headers,
        )
        if response.status_code != 200:
            raise ValueError("Failed to retrieve calibration data from IBM Quantum API. Detailed error: {}".format(response.text))
        self.calibration_json = response.json()

    def _convert_json_to_csv(self, 
                             save_csv:bool=False,
                             destination:str=None,
                             file_name:str=None
                            )->None:
        """
        Private method of the class that converts the raw calibration data in the JSON format obtained from IBM Quantum API to a dataframe and optionally saves it as a CSV file.

        Parameters
        ----------
        save_csv : bool, optional
            Flag to decide whether to save the converted calibration data as a CSV file. Defaults to False.
        desitnation : str, optional
            The directory where the CSV file should be saved if required by the user. If None and the csv needs to be saved, the file will be saved in the current working directory. Defaults to None.
        """
        column_names = ["Qubit", "T1 (us)", "T2 (us)", "Prob meas 0 prep 1", "Prob meas 1 prep 0", "Single Qubit Gate Length (ns)"]
        for gate in self._basis_gates[0]:
            column_names.append(f"{gate} Error")
        column_names.append("Gate Length (ns)")
        for gate in self._basis_gates[1]:
            column_names.append(f"{gate} Gate Length (ns)")
            column_names.append(f"{gate} Error")
        data = pd.DataFrame(columns=column_names)
        add_column_data = {
            "qubits": [],
            "T1": [],
            "T2": [],
            "prob_meas0_prep1": [],
            "prob_meas1_prep0": []
        }
        for qubit, items in enumerate(self.calibration_json["qubits"]):
            add_column_data["qubits"].append(qubit)
            entries = {entry["name"]: entry["value"] for entry in items}
            for name in ("T1", "T2", "prob_meas0_prep1", "prob_meas1_prep0"):
                add_column_data[name].append(entries.get(name, np.nan))
        data["Qubit"] = add_column_data["qubits"]
        t1 = pd.to_numeric(add_column_data["T1"], errors="coerce")
        t2 = pd.to_numeric(add_column_data["T2"], errors="coerce")
        t2_imputed_mask = np.isnan(t2) & ~np.isnan(t1)
        t1_imputed_mask = np.isnan(t1) & ~np.isnan(t2)
        both_missing_mask = np.isnan(t1) & np.isnan(t2)
        t2 = np.where(t2_imputed_mask, t1, t2)
        t1 = np.where(t1_imputed_mask, t2, t1)
        t1 = np.where(both_missing_mask, np.inf, t1)
        t2 = np.where(both_missing_mask, np.inf, t2)
        data["T1 (us)"] = t1
        data["T2 (us)"] = t2
        if both_missing_mask.any():
            print(f"T1 and T2 both unreported, setting to inf (no thermal decay) for qubits: {list(data.index[both_missing_mask])}")
        if t2_imputed_mask.any():
            print(f"Imputed T2 = T1 for qubit(s): {list(data.index[t2_imputed_mask])}")
        if t1_imputed_mask.any():
            print(f"Imputed T1 = T2 for qubit(s): {list(data.index[t1_imputed_mask])}")
        data["Prob meas 0 prep 1"] = add_column_data["prob_meas0_prep1"]
        data["Prob meas 1 prep 0"] = add_column_data["prob_meas1_prep0"]
        for gate in self._basis_gates[1]:
            data.loc[:, "{} Error".format(gate)] = ""
            data.loc[:, "{} Gate Length (ns)".format(gate)] = ""
        for item in self.calibration_json["gates"]:
            if item["gate"] in self._basis_gates[0]:
                col_name = f"{item['gate']} Error"
                error_rate = item["parameters"][0]["value"] if item["parameters"][0]["name"] == "gate_error" else np.nan
                gate_length = item["parameters"][1]["value"] if item["parameters"][1]["name"] == "gate_length" else np.nan
                qubit_index = item["qubits"][0]
                data.loc[qubit_index, col_name] = error_rate
                data.loc[qubit_index, "Single Qubit Gate Length (ns)"] = gate_length
            elif item["gate"] in self._basis_gates[1]:
                col_name = f"{item['gate']} Error"
                control_qubit = item["qubits"][0]
                target_qubit = item["qubits"][1]
                error_rate = item["parameters"][0]["value"] if item["parameters"][0]["name"] == "gate_error" else np.nan
                gate_length = item["parameters"][1]["value"] if item["parameters"][1]["name"] == "gate_length" else np.nan
                gate_length_string = f"{target_qubit}:{gate_length};"
                error_rate_string = f"{target_qubit}:{error_rate};"
                data.loc[control_qubit, f"{item['gate']} Gate Length (ns)"] += gate_length_string
                data.loc[control_qubit, col_name] += error_rate_string
            else:
                continue
        data.loc[:, "Gate Length (ns)"] = ""
        for gate in self._basis_gates[1]:
            for qubit in range(len(data)):
                gate_length_value = data["{} Gate Length (ns)".format(gate)][qubit]
                error_rate_value = data["{} Error".format(gate)][qubit]
                if gate_length_value == "":
                    pass
                else:
                    gate_length_value = gate_length_value[:-1]
                if error_rate_value == "":
                    pass
                else:
                    error_rate_value = error_rate_value[:-1]
                if data.loc[qubit, "Gate Length (ns)"] == "":
                    data.loc[qubit, "Gate Length (ns)"] = gate_length_value
                data.loc[qubit, "{} Error".format(gate)] = error_rate_value
        drop_cols = []
        for gate in self._basis_gates[1]:
            drop_cols.append("{} Gate Length (ns)".format(gate))
        data.drop(columns=drop_cols, inplace=True)
        if save_csv:
            file_path = os.path.join(destination, file_name)
            data.to_csv(file_path, index=False)
        return data

    def get_noise_model(self,
                        save_csv:bool=False,
                        destination:str=None,
                        file_name:str=None
                    )->dict:       
        """
        Method to obtain the noise model in dictionary format from the calibration data of the specified backend. 

        Parameters
        ----------
        save_csv : bool, optional
            Flag to enable saving the converted calibration data as a CSV file. Defaults to False.
        destination : str, optional
            The directory where the CSV file should be saved if required by the used. Defaults to None, in which case the file will be saved in the current working directory. Relevant only if save_csv is set to True.
        file_name : str, optional
            The name of the CSV file to save the converted calibration data. Defaults to None, in which case the file will be saved with a computer generated name. Relevant only if save_csv is set to True.

        Returns
        -------
        dict
            The noise model of the specified backend in dictionary format.
        
        Raises
        ------
        TypeError
            If any of the input arguments are not of the expected type, an error is raised with a detailed message indicating the expected type for each argument.
        NotADirectoryError
            If the specified destination for saving the CSV file is not a valid directory, an error is raised with a detailed message indicating the issue with the provided destination path.
        """
        if not isinstance(save_csv, bool):
            raise TypeError("save_csv must be a boolean value")
        if save_csv:
            if destination is not None:
                if not isinstance(destination, str):
                    raise TypeError("destination must be a string representing the directory to save the CSV file")
                if not os.path.isdir(destination):
                    raise NotADirectoryError("The specified destination is not a directory. Provide a valid path or set destination to None to save in the current working directory.")
            else:
                destination = os.getcwd()
            if file_name is not None:
                if not isinstance(file_name, str):
                    raise TypeError("file_name must be a string")
                if not file_name.endswith(".csv"):
                    file_name += ".csv"
            else:
                time_stamp = time.strftime("%Y%m%d-%H%M%S")
                file_name = f"calibration_data_{self.backend_name}_{time_stamp}.csv"
        self._get_calibration_json()
        calibration_data = self._convert_json_to_csv(save_csv=save_csv, destination=destination, file_name=file_name)
        tmp_file = os.path.join(os.getcwd(), "temp_calibration_data.csv")
        calibration_data.to_csv(tmp_file, index=False)
        noise_model_generator = CreateNoiseModel(
                                                calibration_data_file=tmp_file,
                                                basis_gates=self._basis_gates
                                            )
        noise_model = noise_model_generator.create_noise_model()
        os.remove(tmp_file)
        return noise_model