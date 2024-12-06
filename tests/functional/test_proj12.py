import pickle
from pathlib import Path

from bluepy import Circuit, Simulation

base_path = Path("/gpfs/bbp.cscs.ch/project/proj12/NSE/bluepy/circuits/CA1.O1/mooc-circuit/")
circuit_path = base_path / "CircuitConfig"
simulation_path = base_path / "simulations/sonata_report/BlueConfig"


def test_circuit_pickle():
    circuit = Circuit(circuit_path)
    state = pickle.dumps(circuit)
    initial_state_len = len(state)

    # force caches getting created/filled
    circuit.cells.get()
    circuit.cells.get("Mosaic")
    circuit.projection("SC")

    state = pickle.dumps(circuit)

    assert len(state) == initial_state_len

    new_circuit = pickle.loads(state)
    assert len(circuit.cells.get()) == len(new_circuit.cells.get())


def test_simulation_pickle():
    sim = Simulation(circuit_path)
    state = pickle.dumps(sim)
    initial_state_len = len(state)

    # force caches getting created/filled
    sim.circuit.cells.get()
    sim.circuit.cells.get("Mosaic")
    sim.circuit.projection("SC")

    state = pickle.dumps(sim)

    assert len(state) == initial_state_len

    new_sim = pickle.loads(state)
    assert len(sim.circuit.cells.get()) == len(new_sim.circuit.cells.get())
