# Used for creating new Mutator models from string


from src.servine.evolution.mutator import *
from src.servine.color import fg



class MutatorRegistry:

    _models = {
        "unify": UnifyMutator,
        "nucleotide": NucleotideMutator,
        # TODO add more
    }

    @classmethod
    def get(cls, name, **params):
        model_name = name.lower()

        if model_name in cls._models:
            # Dynamically instantiate the class from the dictionary
            return cls._models[model_name](**params)

        raise ValueError(fg.YELLOW, f"Unknown mutator model: {name}. Available: {list(cls._models.keys())}", fg.RESET)

    @classmethod
    def mutator_params(cls, e_conf):
        mut_params = {}
        # TODO
        #  parse params for mutators, return them for registry params
        mut_params['rate'] = e_conf['mutator']['rate']
        return mut_params







