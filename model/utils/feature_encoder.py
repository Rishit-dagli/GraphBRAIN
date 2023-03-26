import tensorflow as tf
import numpy as np

class FeatureEncoder:
    def __init__(self, allowed_feature_sets):
        self.total_features = 0
        self.feature_mappings = {}
        for feature_name, feature_set in allowed_feature_sets.items():
            sorted_feature_set = sorted(list(feature_set))
            self.feature_mappings[feature_name] = dict(
                zip(
                    sorted_feature_set,
                    range(self.total_features, len(feature_set) + self.total_features),
                )
            )
            self.total_features += len(feature_set)

    def encode(self, inputs):
        output = np.zeros((self.total_features,))
        for feature_name, feature_mapping in self.feature_mappings.items():
            feature_value = getattr(self, feature_name)(inputs)
            if feature_value not in feature_mapping:
                continue
            output[feature_mapping[feature_value]] = 1.0
        return output


class AtomFeatureEncoder(FeatureEncoder):
    def __init__(self, allowed_feature_sets):
        super().__init__(allowed_feature_sets)

    def element(self, atom):
        return atom.GetSymbol()

    def valence_electrons(self, atom):
        return atom.GetTotalValence()

    def hydrogen_bonds(self, atom):
        return atom.GetTotalNumHs()

    def orbital_hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeatureEncoder(FeatureEncoder):
    def __init__(self, allowed_feature_sets):
        super().__init__(allowed_feature_sets)
        self.total_features += 1

    def encode(self, bond):
        output = np.zeros((self.total_features,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugation_state(self, bond):
        return bond.GetIsConjugated()
