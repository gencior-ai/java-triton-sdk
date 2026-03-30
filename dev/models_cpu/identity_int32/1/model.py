import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:

    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            output_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def finalize(self):
        pass
