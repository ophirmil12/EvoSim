# Creates a Sampler from string

from src.servine.io.sampler import (FastaSampler, IdentitySampler, FitnessSampler,
                                    DiversitySampler, PairwiseIdentitySampler, HaplotypeFrequencySampler)
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
        'haplotype': HaplotypeFrequencySampler
    }

    @classmethod
    def get_samplers(cls, conf, **params):
        """Initialize samplers based on configuration."""
        samplers = []
        for s_conf in conf.get('sampling', []):
            sampler_class = cls._samplers.get(s_conf['type'])
            if sampler_class:
                # Special handling for tree which needs initial population size
                if s_conf['type'] == 'tree':
                    samplers.append(sampler_class(
                        s_conf['interval'],
                        s_conf['file'],
                        initial_size=conf['population']['initial_size']
                    ))
                elif s_conf['type'] == 'fasta':
                    samplers.append(sampler_class(
                        s_conf['interval'],
                        s_conf['file'],
                        s_conf['interval']  # backtrack_steps = interval, for fasta
                    ))
                else:
                    samplers.append(sampler_class(s_conf['interval'], s_conf['file']))
            else:
                print(fg.YELLOW, f"Warning: Unknown sampler type '{s_conf['type']}'", fg.RESET)
        return samplers
