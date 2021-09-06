import copy
import math
import numpy as np
"""
To use this: call the train_bm(alphabet, n_states, sequence) function in the global scope.
alphabet should be a list of characters e.g. ['A', 'B', 'C']
The output is 3 numpy arrays to the console: the initial probabilities, transition probabilities and emission probabilities
n_states should be an integer e.g. 5
sequence should be a string using only characters found in the alphabet (this isn't validated so will throw errors on bad characters).
Despite the use of scaling in multiple places to help prevent underflow, very long sequences can still cause underflow as the probabilities become very small.
These were used as inspiration:
https://github.com/hamzarawal/HMM-Baum-Welch-Algorithm/blob/master/baum-welch.py
https://github.com/adeveloperdiary/HiddenMarkovModel/blob/master/part3/BaumWelch.py
"""

class HMM:
    def __init__(self, states, initial, transitions, emissions, alphabet):
        if type(states) is not int:
            raise ValueError('states must be int containing number of model states')
        if type(initial) is not np.ndarray or len(initial.shape) != 1:
            raise ValueError('initial must be 1D numpy array containing initial model probabilities')
        if type(transitions) is not np.ndarray or len(transitions.shape) != 2:
            raise ValueError('transitions must be 2D numpy array containing state transition probabilities')
        if type(emissions) is not np.ndarray or len(emissions.shape) != 2:
            raise ValueError('emissions must be 2D numpy array containing each state\'s character emission probabilities')
        if type(alphabet) is not list:
            raise ValueError('alphabet must be list containing valid characters to emit from a state')
        
        if initial.shape[0] != states:
            raise ValueError('initial probabilities must contain values for all states')
        if transitions.shape[0] != states or transitions.shape[1] != states:
            raise ValueError('transition probabilities must contain values for all states. Matrix must be square')
        if emissions.shape[0] != states or emissions.shape[1] != len(alphabet):
            raise ValueError('emission probabilities must contain values for all states and each state must have all alphabet entries')
        self.states = states
        self.initial = initial
        self.transitions = transitions
        self.emissions = emissions
        self.alphabet = alphabet
        self.alphabet_map = {}
        for i, c in enumerate(self.alphabet):
            self.alphabet_map[c] = i
        self.scale = None

    def forward(self, sequence):
        mat_forward = np.zeros((len(sequence), self.states))
        mat_forward[0, :] = self.initial * self.emissions[:, self.alphabet_map[sequence[0]]]
        scale = np.zeros((len(sequence)))
        scale[0] = np.sum(mat_forward[0, :])
        for i in range(1, len(sequence)):
            for j in range(self.transitions.shape[0]):
                mat_forward[i, j] = np.dot(mat_forward[i - 1], self.transitions[:, j]) * self.emissions[j, self.alphabet_map[sequence[i]]]
            scale[i] = np.sum(mat_forward[i, :])
        for i in range(len(sequence)):
            if scale[i] != 0:
                mat_forward[i, :] /= scale[i]
        sum_scale = np.sum(scale)
        log_likelihood = 0.0
        if sum_scale != 0:
            log_likelihood = np.log(sum_scale)
        self.scale = scale
        return mat_forward, np.sum(mat_forward[-1, :]), log_likelihood

    def backward(self, sequence):
        mat_backward = np.zeros((len(sequence), self.states))
        mat_backward[len(sequence) - 1] = np.ones((self.states))

        for i in range(len(sequence) - 2, -1, -1):
            for j in range(self.states):
                mat_backward[i, j] = np.dot(mat_backward[i + 1] * self.emissions[:, self.alphabet_map[sequence[i + 1]]], self.transitions[j, :])

        for i in range(len(sequence)):
            if self.scale[i] != 0:
                mat_backward[i, :] /= self.scale[i]
        return mat_backward

    def baum_welch(self, sequence):
        mat_f, f_val, l = self.forward(sequence)
        mat_b = self.backward(sequence)
        A = np.zeros((len(sequence), self.states, self.states))
        E = np.zeros((len(sequence), self.states))
        new_initial = self.initial.copy()
        new_transitions = self.transitions.copy()
        new_emissions = self.emissions.copy()

        for i in range(len(sequence)):
            E[i] = (mat_f[i] * mat_b[i]) / f_val
            scale = np.sum(E[i])
            if scale != 0:
                E[i] /= scale
                
        new_initial = E[0]
        for i in range(len(sequence) - 1):
            for j in range(self.states):
                for k in range(self.states):
                    A[i, j, k] = mat_f[i, j] * self.transitions[j, k] * self.emissions[k, self.alphabet_map[sequence[i + 1]]] * mat_b[i + 1, k] / f_val
            scale = np.sum(A[i])
            if scale != 0:
                A[i] /= scale
                
        for i in range(self.states):
            for j in range(self.states):
                numerator = 0.0
                denominator = 0.0
                for k in range(len(sequence) - 1):
                    numerator += A[k, i, j]
                    denominator += E[k, i]
                new_transitions[i, j] = numerator / denominator

        for i in range(self.states):
            for c in range(len(self.alphabet)):
                val = 0.0
                total = 0.0
                for k in range(len(sequence)):
                    if self.alphabet_map[sequence[k]] == c :
                        val += E[k, i]
                    total += E[k, i]
                new_emissions[i, c] = val / total

        return new_initial, new_transitions, new_emissions, l

    def update(self, initial, transitions, emissions):
        self.initial = initial
        self.transitions = transitions
        self.emissions = emissions

    def train_baum_welch(self, sequence, max_iter = 100):
        likelihood = []
        for _ in range(max_iter):
            i, t, e, l = self.baum_welch(sequence)
            likelihood.append(l)
            self.update(i, t, e)
        return likelihood

    def test_common(self, str1, str2):
        if len(str1) != len(str2):
            raise ValueError('string lengths do not match')
        matches = 0
        for i in range(len(str1)):
            if str1[i] == str2[i]:
                matches += 1
        return matches, matches / len(str1)

    def generate(self, sequence_length):
        current_state = np.random.choice(self.states, 1, list(self.initial))[0]
        out = ''
        for i in range(sequence_length):
            chosen_char = np.random.choice(len(self.alphabet), 1, list(self.emissions[current_state]))
            out += self.alphabet[chosen_char[0]]
            current_state = np.random.choice(self.states, 1, list(self.transitions[current_state]))[0]
        return out

    def copy(self):
        return HMM(self.states, self.initial.copy(), self.transitions.copy(), self.emissions.copy(), self.alphabet[:])

    def print(self):
        print(self.initial)
        print(self.transitions)
        print(self.emissions)

class HMMFactory:
    def random_model(self, n_states, alphabet):
        initial = np.random.rand(n_states)
        initial /= initial.sum()
        transitions = np.random.rand(n_states, n_states)
        transitions /= transitions.sum(axis=1)[:,None]
        emissions = np.random.rand(n_states, len(alphabet))
        emissions /= emissions.sum(axis=1)[:,None]
        new_model = HMM(n_states, initial, transitions, emissions, alphabet)
        return new_model

    """
    def generated_baum_welch(self, n_states, alphabet, seq_length = 100, n_sequences = 100):
        trainer = self.random_model(n_states, alphabet)
        generator = self.random_model(n_states, alphabet)
        random = self.random_model(n_states, alphabet)
        log_f0 = 0.0
        log_f1 = 0.0
        log_f2 = 0.0
        genned_seq = []
        for _ in range(n_sequences):
            seq = generator.generate(seq_length)
            genned_seq.append(seq)
            _, _, l = generator.forward(seq)
            log_f0 += l
            i, t, e = trainer.baum_welch(seq)
            trainer.update(i, t, e)
            _, _, l = random.forward(seq)
            log_f2 += l
        for seq in genned_seq:
            _, _, l = trainer.forward(seq)
            log_f1 += l
        genned_seq =[]
        return trainer, generator, abs(log_f0 / n_sequences), abs(log_f1 / n_sequences), abs(log_f2 / n_sequences)
    """

def train_bm(alphabet, n_states, sequence):
    hfac = HMMFactory()
    m = hfac.random_model(n_states, alphabet)
    l = m.train_baum_welch(sequence, 1)
    m.print()

hfac = HMMFactory()
m2 = hfac.random_model(5, ['A', 'B', 'C'])
seq = m2.generate(500)
train_bm(['A', 'B', 'C'], 5, seq)
