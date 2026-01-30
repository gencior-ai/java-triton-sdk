package com.gencior.triton.config;

/**
 * Configuration object for the Triton Inference Server client.
 *
 * <p>This class encapsulates the connection and behavior settings required to communicate with a
 * Triton Inference Server instance. It provides a builder pattern for flexible and fluent
 * configuration.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * TritonClientConfig config = new TritonClientConfig.Builder("http://localhost:8000")
 *     .verbose(true)
 *     .timeout(30000)
 *     .build();
 * }</pre>
 *
 * <p>The configuration supports the following settings:
 * <ul>
 *   <li><strong>URL:</strong> The base URL of the Triton server (required)</li>
 *   <li><strong>Verbose:</strong> Enable verbose logging for debugging purposes (optional, default: false)</li>
 *   <li><strong>Timeout:</strong> Default timeout in milliseconds for client operations (optional, default: 60000ms)</li>
 * </ul>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public class TritonClientConfig {
    private final String url;
    private final boolean verbose;
    private final long defaultTimeoutMs;

    /**
     * Private constructor used by the Builder to create TritonClientConfig instances.
     *
     * @param builder the builder instance containing configuration values
     */
    private TritonClientConfig(Builder builder) {
        this.url = builder.url;
        this.verbose = builder.verbose;
        this.defaultTimeoutMs = builder.defaultTimeoutMs;
    }

    /**
     * Returns the base URL of the Triton Inference Server.
     *
     * @return the Triton server URL (e.g., "http://localhost:8000" or "grpc://localhost:8001")
     */
    public String getUrl() { return url; }

    /**
     * Returns whether verbose logging is enabled for this client.
     *
     * <p>When enabled, the client will produce detailed logging output for debugging purposes.
     *
     * @return {@code true} if verbose logging is enabled, {@code false} otherwise
     */
    public boolean isVerbose() { return verbose; }

    /**
     * Returns the default timeout in milliseconds for client operations.
     *
     * <p>This timeout is used for operations that do not specify their own timeout values.
     * The timeout must be greater than zero.
     *
     * @return the default timeout in milliseconds
     */
    public long getDefaultTimeoutMs() { return defaultTimeoutMs; }

    /**
     * Builder class for constructing {@code TritonClientConfig} instances.
     *
     * <p>This builder follows the fluent builder pattern, allowing method chaining to construct
     * configuration objects with a clean and readable syntax.
     *
     * <h2>Example:</h2>
     * <pre>{@code
     * TritonClientConfig config = new TritonClientConfig.Builder("http://localhost:8000")
     *     .verbose(true)
     *     .timeout(30000)
     *     .build();
     * }</pre>
     *
     * @since 1.0.0
     */
    public static class Builder {
        private String url;
        private boolean verbose = false;
        private long defaultTimeoutMs = 60000;

        /**
         * Creates a new builder for TritonClientConfig.
         *
         * @param url the base URL of the Triton Inference Server (required)
         *            Examples: "http://localhost:8000", "grpc://localhost:8001"
         * @throws IllegalArgumentException if url is null or empty
         */
        public Builder(String url) {
            this.url = url;
        }

        /**
         * Sets whether verbose logging should be enabled for this client.
         *
         * @param verbose {@code true} to enable verbose logging, {@code false} otherwise
         * @return this builder instance for method chaining
         */
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        /**
         * Sets the default timeout for client operations.
         *
         * @param timeoutMs the timeout in milliseconds (must be greater than 0)
         * @return this builder instance for method chaining
         * @throws IllegalArgumentException if timeoutMs is not greater than 0 during build()
         */
        public Builder timeout(long timeoutMs) {
            this.defaultTimeoutMs = timeoutMs;
            return this;
        }

        /**
         * Builds and returns a new {@code TritonClientConfig} instance with the configured settings.
         *
         * @return a new TritonClientConfig instance
         * @throws IllegalArgumentException if the URL is null or empty
         * @throws IllegalArgumentException if the timeout is not greater than 0
         */
        public TritonClientConfig build() {
            if (this.url == null || this.url.isEmpty()) {
                throw new IllegalArgumentException("The Triton server URL is required.");
            }
            if (this.defaultTimeoutMs <= 0) {
                throw new IllegalArgumentException("The timeout must be greater than 0.");
            }
            return new TritonClientConfig(this);
        }
    }
}