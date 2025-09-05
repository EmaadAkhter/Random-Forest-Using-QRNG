import pennylane as qml
import numpy as np
import math


class NumberGenerator:
    def __init__(self, n_feats):
        self.n_feats = n_feats
        self.n_qubits = math.ceil(math.log2(n_feats))
        self.dev = qml.device('default.qubit', wires=self.n_qubits, shots=1)
        self.qnode = qml.QNode(self.circuit, self.dev)

    def circuit(self):
        for bit in range(self.n_qubits):
            qml.Hadamard(wires=bit)
        return qml.sample()

    def generate_unbiased(self):
        while True:
            sample = self.qnode()

            sample = np.asarray(sample)
            if sample.ndim == 0:
                sample = np.array([sample])

            if sample.ndim > 1:
                sample = sample.flatten()

            if len(sample) < self.n_qubits:
                sample = np.pad(sample, (0, self.n_qubits - len(sample)), 'constant')

            sample = sample[:self.n_qubits]

            binary_string = "".join([str(int(bit)) for bit in sample])
            random_int = int(binary_string, 2)
            if random_int < self.n_feats:
                return random_int

    def generate_batch_unbiased(self, batch_size):
        results = []
        while len(results) < batch_size:
            needed = batch_size - len(results)
            batch_device = qml.device('default.qubit', wires=self.n_qubits, shots=needed * 2)
            batch_qnode = qml.QNode(self.circuit, batch_device)

            samples = batch_qnode()

            if samples.ndim == 1 and len(samples) == self.n_qubits:
                samples = samples.reshape(1, -1)
            elif samples.ndim == 1:
                samples = samples.reshape(-1, self.n_qubits)

            for sample in samples:
                if len(results) >= batch_size:
                    break

                sample = np.asarray(sample)
                if sample.ndim == 0:
                    sample = np.array([sample])

                if sample.ndim > 1:
                    sample = sample.flatten()

                if len(sample) < self.n_qubits:
                    sample = np.pad(sample, (0, self.n_qubits - len(sample)), 'constant')

                sample = sample[:self.n_qubits]

                binary_string = "".join([str(int(bit)) for bit in sample])
                random_int = int(binary_string, 2)
                if random_int < self.n_feats:
                    results.append(random_int)

        return results[:batch_size]


def qml_random_choice(n_feats, n_feature, replace=True):
    if n_feats <= 0 or n_feature <= 0:
        return []
    if not replace and n_feature > n_feats:
        raise ValueError("Cannot sample more features than available without replacement.")

    generator = NumberGenerator(n_feats)

    if replace:
        return generator.generate_batch_unbiased(n_feature)
    else:
        chosen_indices = set()
        batch_size = min(n_feature * 2, 100)
        while len(chosen_indices) < n_feature:
            needed = n_feature - len(chosen_indices)
            batch = generator.generate_batch_unbiased(min(batch_size, needed * 2))
            for idx in batch:
                if len(chosen_indices) >= n_feature:
                    break
                chosen_indices.add(idx)
        return list(chosen_indices)[:n_feature]
