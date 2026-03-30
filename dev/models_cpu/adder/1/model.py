import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:

    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            input1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()
            result = (input0 + input1).astype(np.int32)
            output_tensor = pb_utils.Tensor("OUTPUT0", result)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def finalize(self):
        pass
