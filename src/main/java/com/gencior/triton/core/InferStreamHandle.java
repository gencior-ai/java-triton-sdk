package com.gencior.triton.core;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Handle for a streaming inference call, providing lifecycle control.
 *
 * <p>This class is protocol-agnostic and works with both gRPC and HTTP
 * streaming implementations. The cancellation mechanism is injected
 * by the transport layer.
 *
 * <p>Allows the caller to cancel an ongoing stream (e.g., abort LLM token
 * generation mid-response), check completion status, and block until the
 * stream finishes.
 *
 * <p>Implements {@link AutoCloseable} for use in try-with-resources blocks,
 * where closing automatically cancels the stream if it is still running.
 *
 * <h2>Usage — Wait for full completion:</h2>
 * <pre>{@code
 * InferStreamHandle handle = client.inferStream("llm", inputs, listener);
 * handle.await(60, TimeUnit.SECONDS);
 * }</pre>
 *
 * <h2>Usage — Cancel generation early:</h2>
 * <pre>{@code
 * InferStreamHandle handle = client.inferStream("llm", inputs, result -> {
 *     String token = result.asStringArray("text_output")[0];
 * });
 * // Cancel after 5 seconds if still running
 * Thread.sleep(5000);
 * if (!handle.isDone()) {
 *     handle.cancel();
 * }
 * }</pre>
 *
 * <h2>Usage — Auto-close with try-with-resources:</h2>
 * <pre>{@code
 * try (InferStreamHandle handle = client.inferStream("llm", inputs, listener)) {
 *     handle.await(60, TimeUnit.SECONDS);
 * } // stream is cancelled here if still running
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 * @see InferStreamListener
 */
public final class InferStreamHandle implements AutoCloseable {

    private final Runnable cancelAction;
    private final CompletableFuture<Void> completionFuture;

    /**
     * Creates a new stream handle.
     *
     * @param cancelAction the action to execute when cancel is requested
     *                     (e.g., gRPC context cancellation, HTTP request abort)
     * @param completionFuture the future that completes when the stream ends
     */
    public InferStreamHandle(Runnable cancelAction,
                             CompletableFuture<Void> completionFuture) {
        this.cancelAction = cancelAction;
        this.completionFuture = completionFuture;
    }

    /**
     * Cancels the streaming call, aborting token generation on the server.
     *
     * <p>The exact mechanism depends on the transport: gRPC sends a RST_STREAM
     * frame, HTTP aborts the connection. Safe to call multiple times or from
     * any thread.
     */
    public void cancel() {
        cancelAction.run();
    }

    /**
     * Returns whether the stream has finished (completed, errored, or cancelled).
     *
     * @return {@code true} if the stream is done
     */
    public boolean isDone() {
        return completionFuture.isDone();
    }

    /**
     * Blocks until the stream completes.
     *
     * @throws InterruptedException if the current thread is interrupted
     * @throws ExecutionException if the stream completed with an error
     */
    public void await() throws InterruptedException, ExecutionException {
        completionFuture.get();
    }

    /**
     * Blocks until the stream completes or the timeout expires.
     *
     * @param timeout the maximum time to wait
     * @param unit the time unit
     * @throws InterruptedException if the current thread is interrupted
     * @throws ExecutionException if the stream completed with an error
     * @throws TimeoutException if the timeout expired before the stream completed
     */
    public void await(long timeout, TimeUnit unit)
            throws InterruptedException, ExecutionException, TimeoutException {
        completionFuture.get(timeout, unit);
    }

    /**
     * Cancels the stream if still running. Equivalent to {@link #cancel()}.
     */
    @Override
    public void close() {
        cancel();
    }
}
