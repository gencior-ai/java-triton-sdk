package com.gencior.triton.grpc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.gencior.triton.TritonClient;
import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferParameters;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.pojo.TritonModelConfig;
import com.gencior.triton.core.pojo.TritonModelMetadata;
import com.gencior.triton.core.pojo.TritonModelStatistics;
import com.gencior.triton.core.pojo.TritonRepositoryIndex;
import com.gencior.triton.core.pojo.TritonServerMetadata;
import com.gencior.triton.exceptions.TritonDataNotFoundException;

import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceStub;
import inference.GrpcService;
import inference.GrpcService.ModelConfigRequest;
import inference.GrpcService.ModelMetadataRequest;
import inference.GrpcService.ModelReadyRequest;
import inference.GrpcService.ModelStatisticsRequest;
import inference.GrpcService.RepositoryIndexRequest;
import inference.GrpcService.RepositoryModelLoadRequest;
import inference.GrpcService.RepositoryModelUnloadRequest;
import inference.GrpcService.ServerLiveRequest;
import inference.GrpcService.ServerMetadataRequest;
import inference.GrpcService.ServerReadyRequest;
import io.grpc.ChannelCredentials;
import io.grpc.Grpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import io.grpc.TlsChannelCredentials;
import io.grpc.stub.StreamObserver;

/**
 * gRPC-based implementation of the TritonClient for communicating with NVIDIA
 * Triton Inference Server.
 *
 * <p>
 * This class provides a high-performance client implementation using gRPC (gRPC
 * Remote Procedure Call) for synchronous and asynchronous communication with
 * Triton. It handles all aspects of client-server interaction including
 * connection management, request timeout handling, and response parsing.
 *
 * <h2>Features:</h2>
 * <ul>
 * <li><strong>Synchronous Inference:</strong> Blocking inference requests via
 * {@link #infer(String, List)}</li>
 * <li><strong>Asynchronous Inference:</strong> Non-blocking inference with
 * CompletableFuture via {@link #inferAsync(String, List)}</li>
 * <li><strong>Server Monitoring:</strong> Health checks and availability
 * queries</li>
 * <li><strong>Model Management:</strong> Load/unload models, query metadata and
 * statistics</li>
 * <li><strong>Automatic Timeouts:</strong> Configurable per-request timeouts
 * via {@link TritonClientConfig}</li>
 * <li><strong>Error Handling:</strong> Graceful handling of gRPC errors with
 * optional verbose logging</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * TritonClientConfig config = TritonClientConfig.builder()
 *     .url("localhost:8001")
 *     .defaultTimeoutMs(30000)
 *     .verbose(true)
 *     .build();
 *
 * TritonGrpcClient client = new TritonGrpcClient(config);
 * try {
 *     // Check server health
 *     if (client.isServerReady()) {
 *         // Get model metadata
 *         TritonModelMetadata metadata = client.getModelMetadata("my_model");
 *         System.out.println("Model: " + metadata.getName());
 *
 *         // Perform inference
 *         List<InferInput> inputs = Arrays.asList(...);
 *         InferResult result = client.infer("my_model", inputs);
 *         System.out.println("Output: " + result.getOutputAsString("output_0"));
 *     }
 * } finally {
 *     client.close();
 * }
 * }</pre>
 *
 * <h2>Thread Safety:</h2>
 * <p>
 * This client is thread-safe and can be shared across multiple threads. The
 * underlying gRPC channel handles concurrent requests efficiently.
 *
 * <h2>Resource Management:</h2>
 * <p>
 * Always call {@link #close()} to properly release the underlying gRPC channel
 * and cleanup resources. Consider using try-with-resources or try-finally
 * blocks to ensure cleanup.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 * @see TritonClient
 * @see TritonClientConfig
 */
public class TritonGrpcClient implements TritonClient {

    private final TritonClientConfig config;
    private final ManagedChannel channel;
    private final GRPCInferenceServiceBlockingStub blockingStub;
    private final GRPCInferenceServiceStub asyncStub;
    private static final Logger log = LoggerFactory.getLogger(TritonGrpcClient.class);

    /**
     * Creates a new TritonGrpcClient with the given configuration.
     *
     * <p>
     * Initializes a connection to the Triton server specified in the
     * configuration. The connection mode (plaintext or TLS) is determined by
     * {@link TritonClientConfig#isTlsEnabled()}.
     *
     * @param config the client configuration specifying server URL, timeout,
     * TLS settings, and other options
     * @throws io.grpc.StatusRuntimeException if the connection fails
     * @throws IllegalStateException if TLS channel configuration fails
     */
    public TritonGrpcClient(TritonClientConfig config) {
        this.config = config;
        this.channel = buildChannel(config);
        this.blockingStub = GRPCInferenceServiceGrpc.newBlockingStub(channel);
        this.asyncStub = GRPCInferenceServiceGrpc.newStub(channel);
    }

    /**
     * Builds the gRPC managed channel based on the TLS configuration.
     *
     * <p>When TLS is disabled, creates a plaintext channel. When TLS is enabled,
     * configures TLS credentials with optional trust certificate and client certificates
     * for mutual TLS.
     *
     * @param config the client configuration
     * @return the configured ManagedChannel
     * @throws IllegalStateException if TLS channel configuration fails
     */
    private static ManagedChannel buildChannel(TritonClientConfig config) {
        if (!config.isTlsEnabled()) {
            return ManagedChannelBuilder.forTarget(config.getUrl())
                    .usePlaintext()
                    .maxInboundMessageSize(config.getMaxInboundMessageSize())
                    .build();
        }

        try {
            TlsChannelCredentials.Builder tlsBuilder = TlsChannelCredentials.newBuilder();

            if (config.getTrustCertFile() != null) {
                tlsBuilder.trustManager(config.getTrustCertFile());
            }

            if (config.getClientCertFile() != null) {
                tlsBuilder.keyManager(config.getClientCertFile(), config.getClientKeyFile());
            }

            ChannelCredentials creds = tlsBuilder.build();
            return Grpc.newChannelBuilderForAddress(
                            parseHost(config.getUrl()),
                            parsePort(config.getUrl()),
                            creds)
                    .maxInboundMessageSize(config.getMaxInboundMessageSize())
                    .build();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to configure TLS channel", e);
        }
    }

    /**
     * Extracts the host from a target URL string.
     *
     * @param url the target URL (e.g., "localhost:8001" or "dns:///myserver:8001")
     * @return the host portion
     */
    private static String parseHost(String url) {
        String target = url.contains("://") ? url.substring(url.lastIndexOf("://") + 3) : url;
        int colonIdx = target.lastIndexOf(':');
        return colonIdx > 0 ? target.substring(0, colonIdx) : target;
    }

    /**
     * Extracts the port from a target URL string.
     *
     * @param url the target URL (e.g., "localhost:8001" or "dns:///myserver:8001")
     * @return the port number, or 443 as default TLS port
     */
    private static int parsePort(String url) {
        String target = url.contains("://") ? url.substring(url.lastIndexOf("://") + 3) : url;
        int colonIdx = target.lastIndexOf(':');
        if (colonIdx > 0) {
            return Integer.parseInt(target.substring(colonIdx + 1));
        }
        return 443;
    }

    /**
     * Creates a TritonGrpcClient with provided channel and stubs.
     *
     * <p>
     * This constructor is intended for testing purposes and allows injection of
     * mock or custom gRPC stubs.
     *
     * @param config the client configuration
     * @param channel the gRPC managed channel
     * @param blockingStub the blocking inference service stub
     * @param asyncStub the asynchronous inference service stub
     */
    TritonGrpcClient(TritonClientConfig config, ManagedChannel channel,
            GRPCInferenceServiceBlockingStub blockingStub,
            GRPCInferenceServiceStub asyncStub) {
        this.config = config;
        this.channel = channel;
        this.blockingStub = blockingStub;
        this.asyncStub = asyncStub;
    }

    /**
     * Returns the blocking stub with the configured timeout deadline.
     *
     * <p>
     * Each call to this method creates a new stub instance with the deadline
     * applied.
     *
     * @return the blocking stub with timeout deadline
     */
    private GRPCInferenceServiceBlockingStub getStub() {
        return blockingStub.withDeadlineAfter(config.getDefaultTimeoutMs(), TimeUnit.MILLISECONDS);
    }

    /**
     * Executes a gRPC call with automatic timeout and error handling.
     *
     * <p>
     * Wraps gRPC calls to provide consistent error handling and logging. If
     * verbose mode is enabled, errors are logged before being re-thrown.
     *
     * @param <T> the return type of the gRPC call
     * @param operationName the name of the operation (for logging)
     * @param grpcCall the gRPC call to execute
     * @return the result of the gRPC call
     * @throws StatusRuntimeException if the gRPC call fails
     */
    private <T> T executeWithTimeout(String operationName, Supplier<T> grpcCall) {
        long startTime = System.nanoTime();
        try {
            T result = grpcCall.get();
            if (log.isDebugEnabled()) {
                long durationMs = (System.nanoTime() - startTime) / 1_000_000;
                log.debug("gRPC {} completed in {}ms", operationName, durationMs);
            }
            return result;
        } catch (StatusRuntimeException e) {
            long durationMs = (System.nanoTime() - startTime) / 1_000_000;
            log.error("gRPC {} failed after {}ms: {} ({})",
                    operationName, durationMs, e.getStatus().getCode(), e.getMessage());
            throw e;
        }
    }

    /**
     * Checks if the Triton server is alive.
     *
     * <p>
     * This is a lightweight health check that verifies the server process is
     * running. A server can be live but not ready if it's still initializing.
     *
     * @return true if the server is alive, false otherwise
     * @throws StatusRuntimeException if the gRPC call fails
     */
    @Override
    public boolean isServerLive() {
        return this.executeWithTimeout("isServerLive", () -> {
            ServerLiveRequest request = ServerLiveRequest.newBuilder().build();
            return this.getStub().serverLive(request).getLive();
        });
    }

    /**
     * Checks if the Triton server is ready to accept requests.
     *
     * <p>
     * A ready server has completed initialization and is prepared to handle
     * inference requests. This should be checked before attempting to perform
     * inference.
     *
     * @return true if the server is ready, false otherwise
     * @throws StatusRuntimeException if the gRPC call fails
     */
    @Override
    public boolean isServerReady() {
        return this.executeWithTimeout("isServerReady", () -> {
            GrpcService.ServerReadyRequest request = ServerReadyRequest.newBuilder().build();
            return this.getStub().serverReady(request).getReady();
        });
    }

    /**
     * Checks if a specific model is ready to accept inference requests.
     *
     * @param modelId the name of the model to check
     * @param modelVersion the version of the model (can be null for latest
     * version)
     * @return true if the model is ready, false otherwise
     * @throws StatusRuntimeException if the gRPC call fails
     */
    @Override
    public boolean isModelReady(String modelId, String modelVersion) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        return this.executeWithTimeout("isModelReady", () -> {
            GrpcService.ModelReadyRequest.Builder builder = ModelReadyRequest.newBuilder().setName(modelId);
            if (modelVersion != null) {
                builder.setVersion(modelVersion);
            }
            return getStub().modelReady(builder.build()).getReady();
        });
    }

    /**
     * Checks if a specific model is ready to accept inference requests.
     *
     * @param modelId the name of the model to check
     * @return true if the model is ready, false otherwise
     * @throws StatusRuntimeException if the gRPC call fails
     */
    @Override
    public boolean isModelReady(String modelId) {
        return isModelReady(modelId, null);
    }

    /**
     * Retrieves comprehensive metadata about the Triton server.
     *
     * <p>
     * Returns information including server name, version, and supported
     * extensions.
     *
     * @return the server metadata
     * @throws StatusRuntimeException if the gRPC call fails
     * @see TritonServerMetadata
     */
    @Override
    public TritonServerMetadata getServerMetadata() {
        return this.executeWithTimeout("getServerMetadata", () -> {
            GrpcService.ServerMetadataRequest request = ServerMetadataRequest.newBuilder().build();
            return TritonServerMetadata.fromProto(this.getStub().serverMetadata(request));
        });
    }

    /**
     * Retrieves metadata about a specific model's inputs and outputs.
     *
     * <p>
     * The metadata includes tensor names, data types, and shapes for the
     * model's inputs and outputs, which is essential for correctly formatting
     * inference requests.
     *
     * @param modelId the name of the model
     * @param modelVersion the version of the model (can be null for latest
     * version)
     * @return the model metadata including inputs and outputs schema
     * @throws StatusRuntimeException if the gRPC call fails or model not found
     * @see TritonModelMetadata
     */
    @Override
    public TritonModelMetadata getModelMetadata(String modelId, String modelVersion) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        return this.executeWithTimeout("getModelMetadata", () -> {
            GrpcService.ModelMetadataRequest.Builder builder = ModelMetadataRequest.newBuilder().setName(modelId);
            if (modelVersion != null) {
                builder.setVersion(modelVersion);
            }
            return TritonModelMetadata.fromProto(getStub().modelMetadata(builder.build()));
        });
    }

    /**
     * Retrieves runtime configuration information for a specific model.
     *
     * <p>
     * The configuration includes platform type, backend, runtime environment,
     * batching capabilities, and model file mappings.
     *
     * @param modelId the name of the model
     * @param modelVersion the version of the model (can be null for latest
     * version)
     * @return the model runtime configuration
     * @throws StatusRuntimeException if the gRPC call fails or model not found
     * @see TritonModelConfig
     */
    @Override
    public TritonModelConfig getModelConfig(String modelId, String modelVersion) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        return this.executeWithTimeout("getModelConfig", () -> {
            GrpcService.ModelConfigRequest.Builder builder = ModelConfigRequest.newBuilder()
                    .setName(modelId);
            if (modelVersion != null) {
                builder.setVersion(modelVersion);
            }
            return TritonModelConfig.fromProto(this.getStub().modelConfig(builder.build()).getConfig());
        });
    }

    /**
     * Retrieves runtime configuration information for a specific model (latest
     * version).
     *
     * @param modelId the name of the model
     * @return the model runtime configuration
     * @throws StatusRuntimeException if the gRPC call fails or model not found
     * @see TritonModelConfig
     */
    @Override
    public TritonModelConfig getModelConfig(String modelId) {
        return this.getModelConfig(modelId, null);
    }

    /**
     * Retrieves the repository index containing all available models and their
     * status.
     *
     * <p>
     * Returns a listing of all models in the repository, including their names,
     * versions, availability status, and reasons for unavailability if
     * applicable.
     *
     * @return the repository index with all models information
     * @throws StatusRuntimeException if the gRPC call fails
     * @see TritonRepositoryIndex
     */
    @Override
    public TritonRepositoryIndex getModelRepositoryIndex() {
        return this.executeWithTimeout("getModelRepositoryIndex", () -> {
            GrpcService.RepositoryIndexRequest request = RepositoryIndexRequest.newBuilder().build();
            return TritonRepositoryIndex.fromProto(this.getStub().repositoryIndex(request));
        });
    }

    /**
     * Requests the server to load a model.
     *
     * <p>
     * Asynchronously loads the specified model into memory. The model will
     * become available for inference once loading completes. Check model
     * readiness after calling this method.
     *
     * @param modelId the name of the model to load
     * @throws StatusRuntimeException if the gRPC call fails
     */
    @Override
    public void loadModel(String modelId) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        this.executeWithTimeout("loadModel", () -> {
            GrpcService.RepositoryModelLoadRequest request = RepositoryModelLoadRequest.newBuilder().setModelName(modelId).build();
            return this.getStub().repositoryModelLoad(request);
        });
    }

    /**
     * Requests the server to unload a model.
     *
     * <p>
     * Unloads the specified model from memory, freeing associated resources.
     * The model will no longer be available for inference after this call
     * completes.
     *
     * @param modelId the name of the model to unload
     * @throws StatusRuntimeException if the gRPC call fails
     */
    @Override
    public void unLoadModel(String modelId) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        this.executeWithTimeout("unLoadModel", () -> {
            GrpcService.RepositoryModelUnloadRequest request = RepositoryModelUnloadRequest.newBuilder().setModelName(modelId).build();
            return this.getStub().repositoryModelUnload(request);
        });
    }

    /**
     * Retrieves comprehensive inference statistics for a model.
     *
     * <p>
     * Returns performance metrics including inference counts, timing statistics
     * (queue time, compute time, etc.), memory usage, and response statistics.
     * Can query all versions or a specific version.
     *
     * @param modelId the name of the model (can be null to get statistics for
     * all models)
     * @param modelVersion the version of the model (can be null for all
     * versions)
     * @return a list of model statistics objects
     * @throws StatusRuntimeException if the gRPC call fails
     * @see TritonModelStatistics
     */
    @Override
    public List<TritonModelStatistics> getInferenceStatistics(String modelId, String modelVersion) {
        return this.executeWithTimeout("getInferenceStatistics", () -> {
            var builder = ModelStatisticsRequest.newBuilder();
            if (modelId != null) {
                builder.setName(modelId);
            }
            if (modelVersion != null) {
                builder.setVersion(modelVersion);
            }

            return getStub().modelStatistics(builder.build())
                    .getModelStatsList().stream()
                    .map(TritonModelStatistics::fromProto)
                    .toList();
        });
    }

    /**
     * Performs a synchronous (blocking) inference request with custom
     * parameters.
     *
     * <p>
     * This method blocks until the inference result is returned from the server
     * or a timeout occurs. Timeout is controlled via
     * {@link TritonClientConfig#getDefaultTimeoutMs()}.
     *
     * <h2>Input Validation:</h2>
     * <p>
     * All inputs must have raw content available. Inputs are validated to match
     * the model's expected schema (names, data types, shapes) on the server
     * side.
     *
     * @param modelId the name of the model to run inference on
     * @param modelVersion the version of the model (can be null for latest
     * version)
     * @param inputs list of input tensors with data prepared for the model
     * @param customParameters optional map of custom parameters to control
     * inference behavior
     * @return the inference result containing output tensors and response
     * metadata
     * @throws StatusRuntimeException if the gRPC call fails or times out
     * @throws TritonDataNotFoundException if an input lacks raw content
     * @see InferInput
     * @see InferResult
     */
    @Override
    public InferResult infer(
            String modelId,
            String modelVersion,
            List<InferInput> inputs,
            InferParameters customParameters
    ) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(inputs, "inputs must not be null");
        log.debug("infer model={} version={} inputs={}", modelId, modelVersion, inputs.size());
        if (log.isTraceEnabled()) {
            for (InferInput input : inputs) {
                log.trace("  input={} dtype={} shape={} rawBytes={}",
                        input.getName(), input.getDatatype(),
                        java.util.Arrays.toString(input.getShape()),
                        input.hasRawContent() ? input.getRawContent().length : 0);
            }
        }
        return this.executeWithTimeout("infer[" + modelId + "]", () -> {
            var builder = GrpcService.ModelInferRequest.newBuilder()
                    .setModelName(modelId)
                    .setModelVersion(modelVersion != null ? modelVersion : "");

            for (InferInput input : inputs) {
                builder.addInputs(input.getTensor());
                if (input.hasRawContent()) {
                    builder.addRawInputContents(com.google.protobuf.UnsafeByteOperations.unsafeWrap(input.getRawContent()));
                } else {
                    throw new TritonDataNotFoundException(input.getName());
                }
            }
            if (customParameters != null) {
                builder.putAllParameters(toGrpcParameters(customParameters));
            }
            return new InferResult(this.getStub().modelInfer(builder.build()));
        });
    }

    /**
     * Performs a synchronous (blocking) inference request.
     *
     * <p>
     * This method blocks until the inference result is returned from the server
     * or a timeout occurs. Inference is performed on the latest version of the
     * model.
     *
     * @param modelId the name of the model to run inference on
     * @param inputs list of input tensors with data prepared for the model
     * @return the inference result containing output tensors and response
     * metadata
     * @throws StatusRuntimeException if the gRPC call fails or times out
     * @throws TritonDataNotFoundException if an input lacks raw content
     * @see InferResult
     */
    @Override
    public InferResult infer(String modelId, List<InferInput> inputs) {
        return infer(modelId, null, inputs, null);
    }

    /**
     * Performs an asynchronous (non-blocking) inference request with custom
     * parameters.
     *
     * <p>
     * This method returns immediately with a CompletableFuture that will be
     * completed when the inference result is received from the server. The
     * request is executed concurrently in the background. Use the returned
     * future to handle the result or errors.
     *
     * <h2>Error Handling:</h2>
     * <p>
     * Errors can occur during request construction (synchronously) or during
     * server processing (asynchronously). The returned future will be completed
     * exceptionally in case of errors.
     *
     * <h2>Example:</h2>
     * <pre>{@code
     * CompletableFuture<InferResult> future = client.inferAsync(modelId, inputs);
     * future.whenComplete((result, error) -> {
     *     if (error != null) {
     *         System.err.println("Inference failed: " + error.getMessage());
     *     } else {
     *         System.out.println("Result: " + result.getOutputAsString("output_0"));
     *     }
     * });
     * }</pre>
     *
     * @param modelId the name of the model to run inference on
     * @param modelVersion the version of the model (can be null for latest
     * version)
     * @param inputs list of input tensors with data prepared for the model
     * @param customParameters optional map of custom parameters to control
     * inference behavior
     * @return a CompletableFuture that will be completed with the inference
     * result
     * @see InferResult
     */
    @Override
    public CompletableFuture<InferResult> inferAsync(String modelId, String modelVersion, List<InferInput> inputs,
            InferParameters customParameters) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(inputs, "inputs must not be null");
        log.debug("inferAsync model={} version={} inputs={}", modelId, modelVersion, inputs.size());
        final long startTime = System.nanoTime();
        CompletableFuture<InferResult> future = new CompletableFuture<>();
        GrpcService.ModelInferRequest.Builder builder = GrpcService.ModelInferRequest.newBuilder()
                .setModelName(modelId)
                .setModelVersion(modelVersion != null ? modelVersion : "");
        try {
            for (InferInput input : inputs) {
                builder.addInputs(input.getTensor());
                if (input.hasRawContent()) {
                    builder.addRawInputContents(com.google.protobuf.UnsafeByteOperations.unsafeWrap(input.getRawContent()));
                } else {
                    throw new TritonDataNotFoundException(input.getName());
                }
            }
            if (customParameters != null) {
                builder.putAllParameters(toGrpcParameters(customParameters));
            }
        } catch (TritonDataNotFoundException e) {
            future.completeExceptionally(e);
            return future;
        }
        this.asyncStub.modelInfer(builder.build(), new StreamObserver<GrpcService.ModelInferResponse>() {

            @Override
            public void onNext(GrpcService.ModelInferResponse response) {
                long durationMs = (System.nanoTime() - startTime) / 1_000_000;
                log.debug("inferAsync[{}] completed in {}ms", modelId, durationMs);
                future.complete(new InferResult(response));
            }

            @Override
            public void onError(Throwable t) {
                long durationMs = (System.nanoTime() - startTime) / 1_000_000;
                log.error("inferAsync[{}] failed after {}ms: {}", modelId, durationMs, t.getMessage());
                future.completeExceptionally(t);
            }

            @Override
            public void onCompleted() {
                // No-op for unary calls.
                // The result is already processed in onNext().
                // To be implemented for bidirectional streaming.
            }
        });
        return future;
    }

    /**
     * Performs an asynchronous (non-blocking) inference request.
     *
     * <p>
     * This method returns immediately with a CompletableFuture that will be
     * completed when the inference result is received from the server.
     * Inference is performed on the latest version of the model.
     *
     * @param modelId the name of the model to run inference on
     * @param inputs list of input tensors with data prepared for the model
     * @return a CompletableFuture that will be completed with the inference
     * result
     */
    @Override
    public CompletableFuture<InferResult> inferAsync(String modelId, List<InferInput> inputs) {
        return inferAsync(modelId, null, inputs, null);
    }

    /**
     * Closes the client and releases the underlying gRPC channel.
     *
     * <p>
     * This method should be called when the client is no longer needed to free
     * system resources. After calling close(), the client cannot be used for
     * further requests.
     *
     * <p>
     * Attempts to gracefully shutdown the channel with a 5-second timeout. If
     * shutdown doesn't complete within 5 seconds, the channel will be
     * forcefully terminated.
     *
     * @throws Exception if an error occurs during shutdown
     */
    @Override
    public void close() throws Exception {
        if (channel != null && !channel.isShutdown()) {
            channel.shutdown();
            if (!channel.awaitTermination(5, TimeUnit.SECONDS)) {
                log.warn("gRPC channel did not terminate gracefully within 5s, forcing shutdown");
                channel.shutdownNow();
            }
        }
    }

    /**
     * Converts protocol-agnostic {@link InferParameters} to gRPC-specific
     * {@link GrpcService.InferParameter} map.
     *
     * @param params the protocol-agnostic parameters
     * @return a map of gRPC InferParameter values
     */
    private Map<String, GrpcService.InferParameter> toGrpcParameters(InferParameters params) {
        Map<String, GrpcService.InferParameter> grpcParams = new HashMap<>();
        for (Map.Entry<String, Object> entry : params.asMap().entrySet()) {
            GrpcService.InferParameter.Builder paramBuilder = GrpcService.InferParameter.newBuilder();
            Object value = entry.getValue();
            if (value instanceof String s) {
                paramBuilder.setStringParam(s);
            } else if (value instanceof Long l) {
                paramBuilder.setInt64Param(l);
            } else if (value instanceof Boolean b) {
                paramBuilder.setBoolParam(b);
            } else if (value instanceof Double d) {
                paramBuilder.setDoubleParam(d);
            } else {
                throw new IllegalArgumentException(
                        "Unsupported parameter type for key '" + entry.getKey() + "': " + value.getClass().getName());
            }
            grpcParams.put(entry.getKey(), paramBuilder.build());
        }
        return grpcParams;
    }

}
