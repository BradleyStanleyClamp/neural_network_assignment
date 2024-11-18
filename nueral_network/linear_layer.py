from torch import tensor
from .functions import initialize_parameters


class Linear_layer:

    def __init__(self, input_size, output_size) -> None:
        self.w, self.b = initialize_parameters(input_size, output_size)
