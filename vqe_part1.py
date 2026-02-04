import numpy as np
import scipy.optimize
import cirq

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion import get_sparse_operator, jordan_wigner, count_qubits


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


# ------------------ Ansatz ------------------
def create_ansatz(params, qubits, depth):
    circuit = cirq.Circuit()
    idx = 0

    # Initial single-qubit rotations
    for q in qubits:
        circuit.append(cirq.ry(params[idx])(q)); idx += 1
        circuit.append(cirq.rz(params[idx])(q)); idx += 1

    # Entangling layers
    for _ in range(depth):
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        for q in qubits:
            circuit.append(cirq.ry(params[idx])(q)); idx += 1

    return circuit


# ------------------ VQE ------------------
def run_small_vqe(molecule="H2", basis="sto-3g", maxiter=200, depth=2):

    mol = molecule.upper()
    geometry = GEOMETRIES[mol]
    multiplicity = 1
    charge = 0

    moldata = MolecularData(geometry, basis, multiplicity, charge)

    # Run FCI only for H2 (cheap). Skip for LiH to save memory.
    run_fci_flag = (mol == "H2")
    moldata = run_pyscf(moldata, run_scf=True, run_fci=run_fci_flag)

    hf_energy = float(moldata.hf_energy)
    fci_energy = float(moldata.fci_energy) if run_fci_flag else None

    # -------- Active Space to Prevent Memory Crash --------
    if mol == "LIH":
        n_orbitals = moldata.n_orbitals
        active_start = (n_orbitals // 2) - 2
        active_stop = (n_orbitals // 2) + 2

        molecular_hamiltonian = moldata.get_molecular_hamiltonian(
            occupied_indices=range(active_start),
            active_indices=range(active_start, active_stop)
        )
    else:
        molecular_hamiltonian = moldata.get_molecular_hamiltonian()

    # Map to qubit Hamiltonian
    qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)
    sparse_operator = get_sparse_operator(qubit_hamiltonian)

    n_qubits = count_qubits(qubit_hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    num_params = 2 * n_qubits + depth * n_qubits
    simulator = cirq.Simulator()
    energy_history = []

    # -------- Cost Function (Actual Energy Computation) --------
    def cost(params):
        circuit = create_ansatz(params, qubits, depth)
        result = simulator.simulate(circuit)
        psi = result.final_state_vector
        energy = np.vdot(psi, sparse_operator.dot(psi)).real
        energy_history.append(float(energy))
        return energy

    x0 = np.random.uniform(0, 2*np.pi, num_params)

    res = scipy.optimize.minimize(
        cost,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5}
    )

    vqe_energy = float(res.fun)

    # Reference used only for accuracy check
    reference_energy = fci_energy if fci_energy is not None else hf_energy

    return {
        "molecule": mol,
        "n_qubits": n_qubits,
        "hf_energy": round(hf_energy, 6),
        "vqe_energy": round(vqe_energy, 6),
        "reference_energy": round(reference_energy, 6),
        "absolute_error": round(abs(vqe_energy - reference_energy), 6),
        "iterations": len(energy_history),
        "converged": bool(res.success)
    }


# ------------------ Example Run ------------------
if __name__ == "__main__":
    print("Running H2 VQE...")
    print(run_small_vqe("H2"))

    print("\nRunning LiH VQE (Active Space)...")
    print(run_small_vqe("LIH"))
