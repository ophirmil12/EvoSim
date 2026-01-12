# Creates a Sampler from string

from src.servine.io.sampler import (FastaSampler, IdentitySampler, FitnessSampler,
                                    DiversitySampler, PairwiseIdentitySampler, HaplotypeFrequencySampler,
                                    InitialAlleleFrequencySampler, MutationDensitySampler, ZebraHaplotypeSampler)
from src.servine.io.trees import TreeRecorder
from src.servine.color import fg


class SamplerRegistry:
    """Registry for all available samplers."""
    _samplers = {
        'fasta': FastaSampler,
        'identity': IdentitySampler,
        'fitness': FitnessSampler,
        'diversity': DiversitySampler,
        'tree': TreeRecorder,  # Note: computationally intensive - creates a large tree sometimes
        'pairwise': PairwiseIdentitySampler,
        'haplotype': HaplotypeFrequencySampler,
        'initial_alleles': InitialAlleleFrequencySampler,
        'mutation_density': MutationDensitySampler,
        'zebra_haplotype': ZebraHaplotypeSampler
    }

    @classmethod
    def get_samplers(cls, conf, **params):
        """Initialize samplers based on configuration."""
        samplers = []
        for s_conf in conf.get('sampling', []):
            sampler_class = cls._samplers.get(s_conf['type'])

            if not sampler_class:
                continue

            params = cls.sampler_params(s_conf, conf)

            samplers.append(sampler_class(
                interval=s_conf['interval'],
                output_path=s_conf['file'],
                **params
            ))
        return samplers

    @classmethod
    def sampler_params(cls, s_conf, global_conf):
        """
        Similar to MutatorRegistry.mutator_params.
        Handles special requirements for specific samplers.
        """
        # Get the 'params' block from YAML, or empty dict
        params = s_conf.get('params', {}).copy()
        s_type = s_conf['type'].lower()

        # Model-Specific Requirements
        if s_type == 'tree':
            params['initial_size'] = global_conf['population']['initial_size']

        elif s_type == 'fasta':
            # Default backtrack to interval if not specified in YAML
            if 'backtrack_steps' not in params:
                params['backtrack_steps'] = s_conf['interval']

        # Add other specific logic here as you grow the system
        return params