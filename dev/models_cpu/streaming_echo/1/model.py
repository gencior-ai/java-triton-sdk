import time
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Decoupled streaming model that simulates LLM token-by-token generation.

    Splits the input text into words and sends each word as a separate
    streaming response with a small delay between tokens, mimicking
    the behavior of a real LLM backend (vLLM, TensorRT-LLM).
    """

    def initialize(self, args):
        self.decoupled = True

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()

            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_INPUT")
            input_text = input_tensor.as_numpy()[0].decode("utf-8")

            tokens = input_text.split()

            for i, token in enumerate(tokens):
                is_last = (i == len(tokens) - 1)
                output_tensor = pb_utils.Tensor(
                    "TEXT_OUTPUT",
                    np.array([token], dtype=object)
                )
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

                flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if is_last else 0
                response_sender.send(response, flags=flags)

                if not is_last:
                    time.sleep(0.05)

        return None

    def finalize(self):
        pass
