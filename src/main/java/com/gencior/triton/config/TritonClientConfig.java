package com.gencior.triton.config;

import java.io.File;

/**
 * Configuration object for the Triton Inference Server client.
 *
 * <p>This class encapsulates the connection and behavior settings required to communicate with a
 * Triton Inference Server instance. It provides a builder pattern for flexible and fluent
 * configuration.
 *
 * <h2>Plaintext (default):</h2>
 * <pre>{@code
 * TritonClientConfig config = new TritonClientConfig.Builder("localhost:8001")
 *     .timeout(30000)
 *     .build();
 * }</pre>
 *
 * <h2>TLS with server certificate verification:</h2>
 * <pre>{@code
 * TritonClientConfig config = new TritonClientConfig.Builder("localhost:8001")
 *     .tlsEnabled(true)
 *     .trustCertFile(new File("/path/to/ca.pem"))
 *     .build();
 * }</pre>
 *
 * <h2>Mutual TLS (mTLS):</h2>
 * <pre>{@code
 * TritonClientConfig config = new TritonClientConfig.Builder("localhost:8001")
 *     .tlsEnabled(true)
 *     .trustCertFile(new File("/path/to/ca.pem"))
 *     .clientCertFile(new File("/path/to/client.crt"))
 *     .clientKeyFile(new File("/path/to/client.key"))
 *     .build();
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public class TritonClientConfig {
    private final String url;
    private final boolean verbose;
    private final long defaultTimeoutMs;
    private final int maxInboundMessageSize;
    private final boolean tlsEnabled;
    private final File trustCertFile;
    private final File clientCertFile;
    private final File clientKeyFile;

    /**
     * Private constructor used by the Builder to create TritonClientConfig instances.
     *
     * @param builder the builder instance containing configuration values
     */
    private TritonClientConfig(Builder builder) {
        this.url = builder.url;
        this.verbose = builder.verbose;
        this.defaultTimeoutMs = builder.defaultTimeoutMs;
        this.maxInboundMessageSize = builder.maxInboundMessageSize;
        this.tlsEnabled = builder.tlsEnabled;
        this.trustCertFile = builder.trustCertFile;
        this.clientCertFile = builder.clientCertFile;
        this.clientKeyFile = builder.clientKeyFile;
    }

    /**
     * Returns the base URL of the Triton Inference Server.
     *
     * @return the Triton server URL (e.g., "localhost:8001")
     */
    public String getUrl() { return url; }

    /**
     * Returns whether verbose logging is enabled for this client.
     *
     * @return {@code true} if verbose logging is enabled, {@code false} otherwise
     * @deprecated Logging is now managed via SLF4J. Configure the log level for
     * {@code com.gencior.triton} in your logging framework instead
     * (e.g. DEBUG for detailed output, TRACE for tensor shapes and sizes).
     */
    @Deprecated(since = "1.0.0", forRemoval = true)
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
     * Returns the maximum inbound gRPC message size in bytes.
     *
     * <p>This controls the maximum size of responses the client can receive from the server.
     * Defaults to {@link Integer#MAX_VALUE} to allow large batch inference responses.
     *
     * @return the maximum inbound message size in bytes
     */
    public int getMaxInboundMessageSize() { return maxInboundMessageSize; }

    /**
     * Returns whether TLS is enabled for gRPC communication.
     *
     * @return {@code true} if TLS is enabled, {@code false} for plaintext
     */
    public boolean isTlsEnabled() { return tlsEnabled; }

    /**
     * Returns the root CA certificate file for server verification.
     *
     * @return the trust certificate PEM file, or {@code null} to use JVM default truststore
     */
    public File getTrustCertFile() { return trustCertFile; }

    /**
     * Returns the client certificate file for mutual TLS.
     *
     * @return the client certificate PEM file, or {@code null} if mTLS is not configured
     */
    public File getClientCertFile() { return clientCertFile; }

    /**
     * Returns the client private key file for mutual TLS.
     *
     * @return the client private key PEM file, or {@code null} if mTLS is not configured
     */
    public File getClientKeyFile() { return clientKeyFile; }

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
        private int maxInboundMessageSize = Integer.MAX_VALUE;
        private boolean tlsEnabled = false;
        private File trustCertFile;
        private File clientCertFile;
        private File clientKeyFile;

        /**
         * Creates a new builder for TritonClientConfig.
         *
         * @param url the target of the Triton Inference Server (required)
         *            Examples: "localhost:8001", "myserver:8001"
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
         * @deprecated Logging is now managed via SLF4J. Configure the log level for
         * {@code com.gencior.triton} in your logging framework instead.
         */
        @Deprecated(since = "1.0.0", forRemoval = true)
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
         * Sets the maximum inbound gRPC message size in bytes.
         *
         * <p>Defaults to {@link Integer#MAX_VALUE}. Reduce this value if you want to
         * limit the size of responses accepted from the server.
         *
         * @param maxInboundMessageSize the maximum inbound message size in bytes
         * @return this builder instance for method chaining
         */
        public Builder maxInboundMessageSize(int maxInboundMessageSize) {
            this.maxInboundMessageSize = maxInboundMessageSize;
            return this;
        }

        /**
         * Enables or disables TLS for gRPC communication.
         *
         * <p>When enabled, the client uses TLS to encrypt the connection. By default,
         * the JVM's default truststore is used to verify the server certificate. Use
         * {@link #trustCertFile(File)} to provide a custom root CA certificate.
         *
         * @param tlsEnabled {@code true} to enable TLS, {@code false} for plaintext
         * @return this builder instance for method chaining
         */
        public Builder tlsEnabled(boolean tlsEnabled) {
            this.tlsEnabled = tlsEnabled;
            return this;
        }

        /**
         * Sets the root CA certificate file for server verification (PEM format).
         *
         * <p>If not set and TLS is enabled, the JVM's default truststore is used.
         *
         * @param trustCertFile the root CA certificate PEM file
         * @return this builder instance for method chaining
         */
        public Builder trustCertFile(File trustCertFile) {
            this.trustCertFile = trustCertFile;
            return this;
        }

        /**
         * Sets the client certificate file for mutual TLS (PEM format).
         *
         * <p>Must be used together with {@link #clientKeyFile(File)}.
         *
         * @param clientCertFile the client certificate PEM file
         * @return this builder instance for method chaining
         */
        public Builder clientCertFile(File clientCertFile) {
            this.clientCertFile = clientCertFile;
            return this;
        }

        /**
         * Sets the client private key file for mutual TLS (PEM format).
         *
         * <p>Must be used together with {@link #clientCertFile(File)}.
         *
         * @param clientKeyFile the client private key PEM file
         * @return this builder instance for method chaining
         */
        public Builder clientKeyFile(File clientKeyFile) {
            this.clientKeyFile = clientKeyFile;
            return this;
        }

        /**
         * Builds and returns a new {@code TritonClientConfig} instance with the configured settings.
         *
         * @return a new TritonClientConfig instance
         * @throws IllegalArgumentException if the URL is null or empty
         * @throws IllegalArgumentException if the timeout is not greater than 0
         * @throws IllegalArgumentException if certificate files are provided without TLS enabled
         * @throws IllegalArgumentException if only one of clientCertFile/clientKeyFile is provided
         * @throws IllegalArgumentException if certificate files do not exist or are not readable
         */
        public TritonClientConfig build() {
            if (this.url == null || this.url.isEmpty()) {
                throw new IllegalArgumentException("The Triton server URL is required.");
            }
            if (this.defaultTimeoutMs <= 0) {
                throw new IllegalArgumentException("The timeout must be greater than 0.");
            }
            if (this.trustCertFile != null && !this.tlsEnabled) {
                throw new IllegalArgumentException("TLS must be enabled when a trust certificate is provided.");
            }
            if ((this.clientCertFile != null || this.clientKeyFile != null) && !this.tlsEnabled) {
                throw new IllegalArgumentException("TLS must be enabled when client certificates are provided.");
            }
            if ((this.clientCertFile != null) != (this.clientKeyFile != null)) {
                throw new IllegalArgumentException("Both clientCertFile and clientKeyFile must be provided for mutual TLS.");
            }
            if (this.trustCertFile != null) {
                validateFileReadable(this.trustCertFile, "Trust certificate");
            }
            if (this.clientCertFile != null) {
                validateFileReadable(this.clientCertFile, "Client certificate");
                validateFileReadable(this.clientKeyFile, "Client key");
            }
            return new TritonClientConfig(this);
        }

        private void validateFileReadable(File file, String description) {
            if (!file.exists() || !file.isFile() || !file.canRead()) {
                throw new IllegalArgumentException(
                        description + " file not found or not readable: " + file.getAbsolutePath());
            }
        }
    }
}