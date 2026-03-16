package com.gencior.triton.core;

/**
 * Callback interface for consuming streaming inference results token-by-token.
 *
 * <p>Designed for LLM use cases where the server streams back generated tokens
 * incrementally. Only {@link #onToken(InferResult)} is required; the other
 * methods have no-op defaults for convenience.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * client.inferStream("vllm_model", List.of(promptInput), result -> {
 *     System.out.print(result.asStringArray("text_output")[0]);
 * });
 * }</pre>
 *
 * <p><strong>Thread safety:</strong> Callbacks are invoked on gRPC executor threads.
 * Use thread-safe data structures if accumulating results across calls.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 * @see InferStreamHandle
 */
@FunctionalInterface
public interface InferStreamListener {

    /**
     * Called for each streamed inference result (token).
     *
     * @param result the inference result containing one or more output tensors
     */
    void onToken(InferResult result);

    /**
     * Called when the stream completes successfully (all tokens delivered).
     */
    default void onComplete() {}

    /**
     * Called when the stream encounters an error.
     *
     * @param t the error (may be a {@link com.gencior.triton.exceptions.TritonStreamException}
     *          for Triton-level errors, or a {@link io.grpc.StatusRuntimeException} for gRPC errors)
     */
    default void onError(Throwable t) {}
}
