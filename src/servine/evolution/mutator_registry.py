# Used for creating new Mutator models from string


from src.servine.evolution.mutator import *
from src.servine.color import fg


class MutatorRegistry:

    _models = {
        "unify": UnifyMutator,
        "nucleotide": NucleotideMutator,
        "hotcold": HotColdMutator,
        # TODO add more
    }

    _mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    @classmethod
    def get(cls, name, **params):
        model_name = name.lower()

        if model_name in cls._models:
            # Dynamically instantiate the class from the dictionary
            return cls._models[model_name](**params)

        raise ValueError(fg.YELLOW, f"Unknown mutator model: {name}. Available: {list(cls._models.keys())}", fg.RESET)

    @staticmethod
    def _convert_kmers(cls, kmer_list):
        numeric_kmers = []
        for kmer in kmer_list:
            try:
                numeric_kmer = np.array([cls._mapping[base] for base in kmer], dtype=np.uint8)
                numeric_kmers.append(numeric_kmer)
            except KeyError as e:
                raise ValueError(fg.RED, f"Invalid nucleotide in k-mers: {e}", fg.RESET)
        return numeric_kmers

    @classmethod
    def mutator_params(cls, e_conf):
        mut_params = e_conf['mutator'].get('params', {}).copy()
        if 'params' in e_conf['mutator']:
            if 'variable_kmers' in e_conf['mutator']['params']:
                numeric_high_kmers = cls._convert_kmers(
                    cls,
                    e_conf['mutator']['params']['variable_kmers']
                )
                mut_params['variable_kmers'] = numeric_high_kmers
            if 'conserved_kmers' in e_conf['mutator']['params']:
                numeric_low_kmers = cls._convert_kmers(
                    cls,
                    e_conf['mutator']['params']['conserved_kmers']
                )
                mut_params['conserved_kmers'] = numeric_low_kmers

        return mut_params
