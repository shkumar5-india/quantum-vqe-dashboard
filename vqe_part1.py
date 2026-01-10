import numpy as np
import scipy.optimize
import cirq

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion import get_sparse_operator, jordan_wigner, count_qubits


# ----------------------------
# Molecular Geometries
# ----------------------------
GEOMETRIES = {
    "H2": [
        ['H', [0.0, 0.0, 0.0]],
        ['H', [0.0, 0.0, 0.74]]
    ],
    "LIH": [
        ['Li', [0.0, 0.0, 0.0]],
        ['H',  [0.0, 0.0, 1.6]]
    ]
}

REFERENCE = {
    "H2": -1.137,
    "LIH": -7.88
}

# ----------------------------
# VQE Ansatz
# ----------------------------
def create_ansatz(params, qubits, depth=1):
    circuit = cirq.Circuit()
    idx = 0

    # initial rotations
    for q in qubits:
        circuit.append(cirq.ry(params[idx])(q)); idx += 1
        circuit.append(cirq.rz(params[idx])(q)); idx += 1

    # entanglement + layers
    for _ in range(depth):
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        for q in qubits:
            circuit.append(cirq.ry(params[idx])(q)); idx += 1

    return circuit

# ----------------------------
# Part 1 VQE (Real PySCF)
# ----------------------------
def run_small_vqe(molecule="H2", basis="sto-3g", maxiter=60, depth=1):
    mol = molecule.upper()
    if mol not in GEOMETRIES:
        raise ValueError("Supported molecules: H2, LiH")

    geometry = GEOMETRIES[mol]
    multiplicity = 1
    charge = 0

    # MolecularData
    moldata = MolecularData(geometry, basis, multiplicity, charge)

    # PySCF Run
    moldata = run_pyscf(moldata, run_scf=True, run_fci=False)
    hf_energy = float(moldata.hf_energy)

    # Use full Hamiltonian (small molecules only)
    molecular_hamiltonian = moldata.get_molecular_hamiltonian()

    # Map to qubits (Jordan-Wigner)
    qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)
    sparse_operator = get_sparse_operator(qubit_hamiltonian)

    n_qubits = count_qubits(qubit_hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    num_params = 2 * n_qubits + depth * n_qubits

    energy_history = []

    def cost(params):
        circuit = create_ansatz(params, qubits, depth=depth)
        sim = cirq.Simulator()
        result = sim.simulate(circuit)
        psi = result.final_state_vector
        energy = np.vdot(psi, sparse_operator.dot(psi)).real
        energy_history.append(float(energy))
        return energy

    x0 = np.random.uniform(0, 2*np.pi, num_params)
    res = scipy.optimize.minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})

    vqe_energy = float(res.fun)
    ref = REFERENCE.get(mol)

    return {
        "molecule": mol,
        "basis": basis,
        "hf_energy": hf_energy,
        "vqe_energy": vqe_energy,
        "reference_energy": ref,
        "absolute_error": abs(vqe_energy - ref) if ref is not None else None,
        "n_qubits": int(n_qubits),
        "iterations": len(energy_history),
        "energy_history": energy_history,
        "success": bool(res.success)
    }
