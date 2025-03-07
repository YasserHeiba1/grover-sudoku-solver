import numpy as np
import itertools
from math import ceil, log2, floor, pi

from qiskit import QuantumCircuit, Aer, execute

# ------------------------------
# Step 1. Define the 4x4 Sudoku Puzzle (using 0 for empty cells)
puzzle = [
    [4, 3, 0, 2],
    [1, 2, 0, 0],
    [0, 1, 0, 3],
    [0, 0, 0, 0]
]

# ------------------------------
# Step 2. Identify missing positions in the puzzle.
def get_missing_positions(puzzle):
    missing = []
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] == 0:
                missing.append((i, j))
    return missing

missing_positions = get_missing_positions(puzzle)
num_missing = len(missing_positions)

# ------------------------------
# Step 3. Enumerate candidates for the missing cells.
def fill_candidate(puzzle, missing_positions, candidate):
    new_puzzle = [row.copy() for row in puzzle]
    for (i, j), val in zip(missing_positions, candidate):
        new_puzzle[i][j] = val
    return new_puzzle

def is_valid_sudoku(puzzle):
    n = 4
    block_size = 2
    # Check rows
    for row in puzzle:
        if sorted(row) != list(range(1, n + 1)):
            return False
    # Check columns
    for j in range(n):
        col = [puzzle[i][j] for i in range(n)]
        if sorted(col) != list(range(1, n + 1)):
            return False
    # Check 2x2 subgrids
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            block = []
            for di in range(block_size):
                for dj in range(block_size):
                    block.append(puzzle[i + di][j + dj])
            if sorted(block) != list(range(1, n + 1)):
                return False
    return True

candidate_list = list(itertools.product(range(1, 5), repeat=num_missing))
valid_candidate_index = None
for idx, candidate in enumerate(candidate_list):
    candidate_puzzle = fill_candidate(puzzle, missing_positions, candidate)
    if is_valid_sudoku(candidate_puzzle):
        valid_candidate_index = idx
        break

if valid_candidate_index is None:
    print("No valid candidate found!")
    exit()

# ------------------------------
# Step 4. Map candidate list to a quantum register.
num_candidates = len(candidate_list)
n_qubits = ceil(log2(num_candidates))

# ------------------------------
# Step 5. Construct the Oracle to flip the phase of the valid candidate.
def construct_oracle(n_qubits, valid_index):
    oracle = QuantumCircuit(n_qubits)
    # Convert valid_index to binary with n_qubits bits.
    bin_string = format(valid_index, f'0{n_qubits}b')
    # Prepare for multi-controlled gate by flipping bits that are 0.
    for i, bit in enumerate(reversed(bin_string)):
        if bit == '0':
            oracle.x(i)
    # Apply multi-controlled Z using an MCX (with an H before and after).
    oracle.h(n_qubits - 1)
    oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    oracle.h(n_qubits - 1)
    # Undo the initial X gates.
    for i, bit in enumerate(reversed(bin_string)):
        if bit == '0':
            oracle.x(i)
    oracle.name = "Oracle"
    return oracle

oracle = construct_oracle(n_qubits, valid_candidate_index)

# ------------------------------
# Step 6. Build the Diffusion Operator.
def diffusion_operator(n_qubits):
    diffuser = QuantumCircuit(n_qubits)
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    diffuser.name = "Diffuser"
    return diffuser

diffuser = diffusion_operator(n_qubits)

# ------------------------------
# Step 7. Determine the number of Grover iterations.
iterations = int(np.floor(pi / 4 * np.sqrt(num_candidates)))

# ------------------------------
# Step 8. Assemble the Grover Circuit.
grover = QuantumCircuit(n_qubits, n_qubits)
grover.h(range(n_qubits))
for _ in range(iterations):
    grover.append(oracle.to_gate(), range(n_qubits))
    grover.append(diffuser.to_gate(), range(n_qubits))
grover.measure(range(n_qubits), range(n_qubits))

# ------------------------------
# Step 9. Execute the Circuit on a Simulator.
backend = Aer.get_backend('qasm_simulator')
result = execute(grover, backend, shots=1024).result()
counts = result.get_counts()

# Decode the most frequent measurement to get the candidate index.
most_common = max(counts, key=counts.get)
found_index = int(most_common, 2)

# ------------------------------
# Step 10. Construct the final solution and output it.
solution_puzzle = fill_candidate(puzzle, missing_positions, candidate_list[found_index])

# Output only the valid solution.
print("Valid 4x4 Sudoku solution:")
for row in solution_puzzle:
    print(row)
