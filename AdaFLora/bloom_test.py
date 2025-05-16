from transformers import PretrainedConfig

class BloomConfig(PretrainedConfig):
    model_type = "bloom"

    def __init__(self,
                 