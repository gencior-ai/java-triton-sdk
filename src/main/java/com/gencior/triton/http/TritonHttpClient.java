package com.gencior.triton.http;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;
import java.util.concurrent.Flow.Publisher;
import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.gencior.triton.TritonClient;
import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferParameters;
import com.gencior.triton.core.InferRequestedOutput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.InferResult.OutputTensorDescriptor;
import com.gencior.triton.core.InferStreamHandle;
import com.gencior.triton.core.InferStreamListener;
import com.gencior.triton.core.TritonDataType;
import com.gencior.triton.core.pojo.TritonModelConfig;
import com.gencior.triton.core.pojo.TritonModelMetadata;
import com.gencior.triton.core.pojo.TritonModelStatistics;
import com.gencior.triton.core.pojo.TritonRepositoryIndex;
import com.gencior.triton.core.pojo.TritonServerMetadata;
import com.gencior.triton.exceptions.TritonInferException;

/**
 * HTTP/REST-based implementation of the {@link TritonClient} interface for communicating
 * with NVIDIA Triton Inference Server.
 *
 * <p>This client uses the Triton v2 HTTP/REST inference protocol and the built-in
 * {@link java.net.http.HttpClient} (Java 11+). It maintains identical method signatures
 * as the gRPC implementation for seamless switching between transport protocols.</p>
 *
 * <h2>Features:</h2>
 * <ul>
 *   <li><strong>Server Monitoring:</strong> Health checks and metadata queries via
 *       {@link #isServerLive()}, {@link #isServerReady()}, {@link #getServerMetadata()}</li>
 *   <li><strong>Model Management:</strong> Load/unload models, query metadata, config,
 *       statistics, and repository index</li>
 *   <li><strong>TLS/mTLS Support:</strong> Configurable via {@link TritonClientConfig}
 *       with PEM certificate files</li>
 *   <li><strong>Automatic Timeouts:</strong> Per-request timeouts from
 *       {@link TritonClientConfig#getDefaultTimeoutMs()}</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * TritonClientConfig config = new TritonClientConfig.Builder("localhost:8000")
 *         .timeout(30000)
 *         .build();
 *
 * try (TritonHttpClient client = new TritonHttpClient(config)) {
 *     if (client.isServerReady()) {
 *         TritonServerMetadata metadata = client.getServerMetadata();
 *         System.out.println("Server: " + metadata.getName());
 *     }
 * }
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.1.0
 * @see TritonClient
 * @see TritonClientConfig
 */
public class TritonHttpClient implements TritonClient {

    private static final Logger LOG = LoggerFactory.getLogger(TritonHttpClient.class);

    private final TritonHttpHelper helper;

    /**
     * Creates a new HTTP client for the Triton Inference Server.
     *
     * <p>The client establishes an internal {@link java.net.http.HttpClient} configured
     * with the timeout and TLS settings from the provided configuration. If TLS is enabled,
     * an {@link javax.net.ssl.SSLContext} is created from the configured PEM certificate files.</p>
     *
     * @param config the client configuration containing URL, timeout, and optional TLS settings
     * @throws TritonInferException if SSL context creation fails when TLS is enabled
     */
    public TritonHttpClient(TritonClientConfig config) {
        this.helper = new TritonHttpHelper(config);
        LOG.info("TritonHttpClient initialized for {}", helper.getBaseUrl());
    }

    /**
     * Closes this HTTP client and releases associated resources.
     *
     * <p>The underlying {@link java.net.http.HttpClient} does not require explicit shutdown,
     * so this method is effectively a no-op. It is safe to call multiple times.</p>
     *
     * @throws Exception never thrown by this implementation
     */
    @Override
    public void close() throws Exception {
        LOG.debug("TritonHttpClient closed");
    }

    /**
     * Checks if the Triton server process is alive.
     *
     * <p>Sends a {@code GET /v2/health/live} request. Returns {@code true} if the server
     * responds with HTTP 200, {@code false} otherwise (including network errors).</p>
     *
     * @return {@code true} if the server is live, {@code false} otherwise
     */
    @Override
    public boolean isServerLive() {
        int status = helper.sendGetStatus("/v2/health/live");
        return status == 200;
    }

    /**
     * Checks if the Triton server is ready to accept inference requests.
     *
     * <p>Sends a {@code GET /v2/health/ready} request. A server may be live but not ready
     * if it is still loading models or initializing backends.</p>
     *
     * @return {@code true} if the server is ready, {@code false} otherwise
     */
    @Override
    public boolean isServerReady() {
        int status = helper.sendGetStatus("/v2/health/ready");
        return status == 200;
    }

    /**
     * Retrieves metadata about the Triton server instance.
     *
     * <p>Sends a {@code GET /v2} request and returns the server name, version,
     * and supported extensions.</p>
     *
     * @return the server metadata, never {@code null}
     * @throws TritonInferException if the request fails or the response cannot be parsed
     */
    @Override
    public TritonServerMetadata getServerMetadata() {
        JsonNode json = helper.sendGet("/v2");
        return TritonServerMetadata.fromJson(json);
    }

    /**
     * Checks if a model (latest version) is ready for inference.
     *
     * <p>Sends a {@code GET /v2/models/{modelId}/ready} request.</p>
     *
     * @param modelId the name of the model to check
     * @return {@code true} if the model is loaded and ready, {@code false} otherwise
     */
    @Override
    public boolean isModelReady(String modelId) {
        int status = helper.sendGetStatus("/v2/models/" + modelId + "/ready");
        return status == 200;
    }

    /**
     * Checks if a specific version of a model is ready for inference.
     *
     * <p>Sends a {@code GET /v2/models/{modelId}/versions/{modelVersion}/ready} request.</p>
     *
     * @param modelId the name of the model to check
     * @param modelVersion the specific version to check
     * @return {@code true} if the model version is loaded and ready, {@code false} otherwise
     */
    @Override
    public boolean isModelReady(String modelId, String modelVersion) {
        int status = helper.sendGetStatus("/v2/models/" + modelId + "/versions/" + modelVersion + "/ready");
        return status == 200;
    }

    /**
     * Retrieves metadata for a specific model version, including input/output tensor schemas.
     *
     * <p>Sends a {@code GET /v2/models/{modelId}/versions/{modelVersion}} request.
     * If {@code modelVersion} is {@code null} or empty, the latest version metadata is returned.</p>
     *
     * @param modelId the name of the model
     * @param modelVersion the model version, or {@code null}/empty for the latest version
     * @return the model metadata containing name, platform, inputs, and outputs
     * @throws TritonInferException if the model is not found or the request fails
     */
    @Override
    public TritonModelMetadata getModelMetadata(String modelId, String modelVersion) {
        String path = "/v2/models/" + modelId;
        if (modelVersion != null && !modelVersion.isEmpty()) {
            path += "/versions/" + modelVersion;
        }
        JsonNode json = helper.sendGet(path);
        return TritonModelMetadata.fromJson(json);
    }

    /**
     * Retrieves the runtime configuration of a model (latest version).
     *
     * <p>Sends a {@code GET /v2/models/{modelId}/config} request. The configuration
     * includes the backend, platform, batch size, and model file details.</p>
     *
     * @param modelId the name of the model
     * @return the model configuration
     * @throws TritonInferException if the model is not found or the request fails
     */
    @Override
    public TritonModelConfig getModelConfig(String modelId) {
        JsonNode json = helper.sendGet("/v2/models/" + modelId + "/config");
        return TritonModelConfig.fromJson(json);
    }

    /**
     * Retrieves the runtime configuration of a specific model version.
     *
     * <p>Sends a {@code GET /v2/models/{modelId}/versions/{modelVersion}/config} request.
     * If {@code modelVersion} is {@code null} or empty, the latest version config is returned.</p>
     *
     * @param modelId the name of the model
     * @param modelVersion the model version, or {@code null}/empty for the latest version
     * @return the model configuration
     * @throws TritonInferException if the model is not found or the request fails
     */
    @Override
    public TritonModelConfig getModelConfig(String modelId, String modelVersion) {
        String path = "/v2/models/" + modelId;
        if (modelVersion != null && !modelVersion.isEmpty()) {
            path += "/versions/" + modelVersion;
        }
        path += "/config";
        JsonNode json = helper.sendGet(path);
        return TritonModelConfig.fromJson(json);
    }

    /**
     * Retrieves the index of all models available in the server's model repository.
     *
     * <p>Sends a {@code POST /v2/repository/index} request with an empty body.
     * The index includes model names, versions, and their current state
     * (READY, UNAVAILABLE, LOADING, UNLOADING).</p>
     *
     * @return the repository index containing all model entries
     * @throws TritonInferException if the request fails
     */
    @Override
    public TritonRepositoryIndex getModelRepositoryIndex() {
        JsonNode json = helper.sendPostEmpty("/v2/repository/index");
        return TritonRepositoryIndex.fromJson(json);
    }

    /**
     * Requests the server to load a model into memory.
     *
     * <p>Sends a {@code POST /v2/repository/models/{modelId}/load} request.
     * The model must exist in the server's model repository. After this call returns
     * successfully, the model should become ready for inference (use {@link #isModelReady(String)}
     * to verify).</p>
     *
     * @param modelId the name of the model to load
     * @throws TritonInferException if the model cannot be loaded or is not found in the repository
     */
    @Override
    public void loadModel(String modelId) {
        helper.sendPostEmptyVoid("/v2/repository/models/" + modelId + "/load");
        LOG.debug("Model '{}' load requested", modelId);
    }

    /**
     * Requests the server to unload a model from memory.
     *
     * <p>Sends a {@code POST /v2/repository/models/{modelId}/unload} request.
     * After this call, the model will no longer be available for inference.</p>
     *
     * @param modelId the name of the model to unload
     * @throws TritonInferException if the request fails
     */
    @Override
    public void unLoadModel(String modelId) {
        helper.sendPostEmptyVoid("/v2/repository/models/" + modelId + "/unload");
        LOG.debug("Model '{}' unload requested", modelId);
    }

    /**
     * Retrieves inference performance statistics for a specific model version.
     *
     * <p>Sends a {@code GET /v2/models/{modelId}/versions/{modelVersion}/stats} request.
     * Statistics include inference counts, timing breakdowns (queue, compute, output),
     * success/failure rates, and memory usage.</p>
     *
     * @param modelId the name of the model
     * @param modelVersion the model version, or {@code null}/empty for all versions
     * @return a list of statistics for each matching model version
     * @throws TritonInferException if the model is not found or the request fails
     */
    @Override
    public List<TritonModelStatistics> getInferenceStatistics(String modelId, String modelVersion) {
        String path = "/v2/models/" + modelId;
        if (modelVersion != null && !modelVersion.isEmpty()) {
            path += "/versions/" + modelVersion;
        }
        path += "/stats";
        JsonNode json = helper.sendGet(path);
        List<TritonModelStatistics> stats = new ArrayList<>();
        JsonNode modelStats = json.path("model_stats");
        if (modelStats.isArray()) {
            for (JsonNode stat : modelStats) {
                stats.add(TritonModelStatistics.fromJson(stat));
            }
        }
        return stats;
    }

    /** {@inheritDoc} */
    @Override
    public InferResult infer(String modelId, List<InferInput> inputs) {
        return infer(modelId, null, inputs, null, null);
    }

    /** {@inheritDoc} */
    @Override
    public InferResult infer(String modelId, String modelVersion, List<InferInput> inputs,
            InferParameters customParameters) {
        return infer(modelId, modelVersion, inputs, null, customParameters);
    }

    /** {@inheritDoc} */
    @Override
    public InferResult infer(String modelId, String modelVersion, List<InferInput> inputs,
            List<InferRequestedOutput> outputs, InferParameters customParameters) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(inputs, "inputs must not be null");

        String path = buildInferPath(modelId, modelVersion);
        String jsonHeader = buildInferRequestJson(inputs, outputs, customParameters);
        byte[] binaryData = concatenateRawContent(inputs);

        LOG.debug("infer[{}] inputs={} outputs={}", modelId, inputs.size(),
                outputs != null ? outputs.size() : "all");

        HttpResponse<byte[]> response = helper.sendPostBinary(path, jsonHeader, binaryData);
        return parseInferResponse(response, modelId, modelVersion);
    }


    /** {@inheritDoc} */
    @Override
    public CompletableFuture<InferResult> inferAsync(String modelId, List<InferInput> inputs) {
        return inferAsync(modelId, null, inputs, null, null);
    }

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<InferResult> inferAsync(String modelId, String modelVersion, List<InferInput> inputs,
            InferParameters customParameters) {
        return inferAsync(modelId, modelVersion, inputs, null, customParameters);
    }

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<InferResult> inferAsync(String modelId, String modelVersion, List<InferInput> inputs,
            List<InferRequestedOutput> outputs, InferParameters customParameters) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(inputs, "inputs must not be null");

        return CompletableFuture.supplyAsync(() ->
                infer(modelId, modelVersion, inputs, outputs, customParameters));
    }

    /** {@inheritDoc} */
    @Override
    public InferStreamHandle inferStream(String modelId, List<InferInput> inputs, InferStreamListener listener) {
        return inferStream(modelId, null, inputs, null, null, listener);
    }

    /** {@inheritDoc} */
    @Override
    public InferStreamHandle inferStream(String modelId, String modelVersion, List<InferInput> inputs,
            InferParameters customParameters, InferStreamListener listener) {
        return inferStream(modelId, modelVersion, inputs, null, customParameters, listener);
    }

    /** {@inheritDoc} */
    @Override
    public InferStreamHandle inferStream(String modelId, String modelVersion, List<InferInput> inputs,
            List<InferRequestedOutput> outputs, InferParameters customParameters, InferStreamListener listener) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(listener, "listener must not be null");

        String path = buildStreamPath(modelId, modelVersion);
        String jsonBody = buildGenerateRequestJson(inputs, customParameters);
        CompletableFuture<Void> completionFuture = new CompletableFuture<>();
        AtomicBoolean cancelled = new AtomicBoolean(false);

        Thread streamThread = Thread.ofVirtual().name("triton-sse-" + modelId).start(() -> {
            try {
                HttpResponse<InputStream> response = helper.sendPostStream(path, jsonBody);
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(response.body(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null && !cancelled.get()) {
                        if (line.startsWith("data:")) {
                            String data = line.substring(5).trim();
                            if (data.isEmpty()) continue;
                            InferResult result = parseSseGenerateEvent(data, modelId, modelVersion);
                            listener.onToken(result);
                        }
                    }
                }
                if (!cancelled.get()) {
                    listener.onComplete();
                    completionFuture.complete(null);
                }
            } catch (Exception e) {
                if (!cancelled.get()) {
                    listener.onError(e);
                    completionFuture.completeExceptionally(e);
                }
            }
        });

        return new InferStreamHandle(() -> {
            cancelled.set(true);
            streamThread.interrupt();
        }, completionFuture);
    }

    /** {@inheritDoc} */
    @Override
    public Flow.Publisher<InferResult> inferStreamPublisher(String modelId, List<InferInput> inputs) {
        return inferStreamPublisher(modelId, null, inputs, null, null);
    }

    /** {@inheritDoc} */
    @Override
    public Flow.Publisher<InferResult> inferStreamPublisher(String modelId, String modelVersion,
            List<InferInput> inputs, InferParameters customParameters) {
        return inferStreamPublisher(modelId, modelVersion, inputs, null, customParameters);
    }

    /** {@inheritDoc} */
    @Override
    public Publisher<InferResult> inferStreamPublisher(String modelId, String modelVersion, List<InferInput> inputs,
            List<InferRequestedOutput> outputs, InferParameters customParameters) {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(inputs, "inputs must not be null");

        return subscriber -> {
            Objects.requireNonNull(subscriber, "subscriber must not be null");
            AtomicBoolean cancelled = new AtomicBoolean(false);

            subscriber.onSubscribe(new Flow.Subscription() {
                @Override
                public void request(long n) {
                    if (n <= 0) {
                        subscriber.onError(new IllegalArgumentException(
                                "Flow.Subscription.request: n must be > 0, got " + n));
                    }
                }

                @Override
                public void cancel() {
                    cancelled.set(true);
                }
            });

            String path = buildStreamPath(modelId, modelVersion);
            String jsonBody = buildGenerateRequestJson(inputs, customParameters);

            Thread.ofVirtual().name("triton-sse-pub-" + modelId).start(() -> {
                try {
                    HttpResponse<InputStream> response = helper.sendPostStream(path, jsonBody);
                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(response.body(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null && !cancelled.get()) {
                            if (line.startsWith("data:")) {
                                String data = line.substring(5).trim();
                                if (data.isEmpty()) continue;
                                InferResult result = parseSseGenerateEvent(data, modelId, modelVersion);
                                subscriber.onNext(result);
                            }
                        }
                    }
                    if (!cancelled.get()) {
                        subscriber.onComplete();
                    }
                } catch (Exception e) {
                    if (!cancelled.get()) {
                        subscriber.onError(e);
                    }
                }
            });
        };
    }

    // ========== Internal: Request Building ==========

    private String buildInferPath(String modelId, String modelVersion) {
        String path = "/v2/models/" + modelId;
        if (modelVersion != null && !modelVersion.isEmpty()) {
            path += "/versions/" + modelVersion;
        }
        return path + "/infer";
    }

    private String buildStreamPath(String modelId, String modelVersion) {
        String path = "/v2/models/" + modelId;
        if (modelVersion != null && !modelVersion.isEmpty()) {
            path += "/versions/" + modelVersion;
        }
        return path + "/generate_stream";
    }

    /**
     * Builds a flat JSON request for the Triton generate/generate_stream endpoint.
     * The generate endpoint expects inputs as top-level keys (not the v2 infer format).
     *
     * <p>Example output: {@code {"TEXT_INPUT": "hello world", "param1": "value1"}}</p>
     */
    private String buildGenerateRequestJson(List<InferInput> inputs, InferParameters customParameters) {
        ObjectMapper mapper = helper.getObjectMapper();
        ObjectNode root = mapper.createObjectNode();

        for (InferInput input : inputs) {
            if (!input.hasRawContent()) continue;
            TritonDataType dtype = input.getDatatype();
            if (dtype == TritonDataType.BYTES) {
                // Deserialize length-prefixed strings back to a JSON value
                String[] strings = input.getDataAsStringArray();
                if (strings.length == 1) {
                    root.put(input.getName(), strings[0]);
                } else {
                    ArrayNode arr = root.putArray(input.getName());
                    for (String s : strings) arr.add(s);
                }
            } else if (dtype == TritonDataType.FP32) {
                float[] data = input.getDataAsFloatArray();
                if (data.length == 1) {
                    root.put(input.getName(), data[0]);
                } else {
                    ArrayNode arr = root.putArray(input.getName());
                    for (float v : data) arr.add(v);
                }
            } else if (dtype == TritonDataType.INT32) {
                int[] data = input.getDataAsIntArray();
                if (data.length == 1) {
                    root.put(input.getName(), data[0]);
                } else {
                    ArrayNode arr = root.putArray(input.getName());
                    for (int v : data) arr.add(v);
                }
            } else if (dtype == TritonDataType.INT64 || dtype == TritonDataType.UINT64) {
                long[] data = input.getDataAsLongArray();
                if (data.length == 1) {
                    root.put(input.getName(), data[0]);
                } else {
                    ArrayNode arr = root.putArray(input.getName());
                    for (long v : data) arr.add(v);
                }
            } else if (dtype == TritonDataType.FP64) {
                double[] data = input.getDataAsDoubleArray();
                if (data.length == 1) {
                    root.put(input.getName(), data[0]);
                } else {
                    ArrayNode arr = root.putArray(input.getName());
                    for (double v : data) arr.add(v);
                }
            } else if (dtype == TritonDataType.BOOL) {
                boolean[] data = input.getDataAsBooleanArray();
                if (data.length == 1) {
                    root.put(input.getName(), data[0]);
                } else {
                    ArrayNode arr = root.putArray(input.getName());
                    for (boolean v : data) arr.add(v);
                }
            }
        }

        if (customParameters != null) {
            for (Map.Entry<String, Object> entry : customParameters.asMap().entrySet()) {
                addParameterValue(root, entry.getKey(), entry.getValue());
            }
        }

        try {
            return mapper.writeValueAsString(root);
        } catch (JsonProcessingException e) {
            throw new TritonInferException("Failed to build generate request JSON", e);
        }
    }

    private String buildInferRequestJson(List<InferInput> inputs,
            List<InferRequestedOutput> outputs, InferParameters customParameters) {
        ObjectMapper mapper = helper.getObjectMapper();
        ObjectNode root = mapper.createObjectNode();

        ArrayNode inputsArray = root.putArray("inputs");
        for (InferInput input : inputs) {
            ObjectNode inputNode = mapper.createObjectNode();
            inputNode.put("name", input.getName());
            inputNode.put("datatype", input.getDatatypeString());
            ArrayNode shapeArray = inputNode.putArray("shape");
            for (long dim : input.getShape()) {
                shapeArray.add(dim);
            }
            ObjectNode params = inputNode.putObject("parameters");
            params.put("binary_data_size", input.hasRawContent() ? input.getRawContent().length : 0);
            inputsArray.add(inputNode);
        }

        if (outputs != null && !outputs.isEmpty()) {
            ArrayNode outputsArray = root.putArray("outputs");
            for (InferRequestedOutput output : outputs) {
                ObjectNode outputNode = mapper.createObjectNode();
                outputNode.put("name", output.getName());
                ObjectNode outParams = outputNode.putObject("parameters");
                outParams.put("binary_data", true);
                if (output.hasParameters()) {
                    for (Map.Entry<String, Object> entry : output.getParameters().entrySet()) {
                        addParameterValue(outParams, entry.getKey(), entry.getValue());
                    }
                }
                outputsArray.add(outputNode);
            }
        }

        if (customParameters != null) {
            ObjectNode paramsNode = root.putObject("parameters");
            for (Map.Entry<String, Object> entry : customParameters.asMap().entrySet()) {
                addParameterValue(paramsNode, entry.getKey(), entry.getValue());
            }
        }

        try {
            return mapper.writeValueAsString(root);
        } catch (JsonProcessingException e) {
            throw new TritonInferException("Failed to build inference request JSON", e);
        }
    }

    private void addParameterValue(ObjectNode node, String key, Object value) {
        if (value instanceof String s) {
            node.put(key, s);
        } else if (value instanceof Long l) {
            node.put(key, l);
        } else if (value instanceof Boolean b) {
            node.put(key, b);
        } else if (value instanceof Double d) {
            node.put(key, d);
        } else if (value instanceof Integer i) {
            node.put(key, i);
        }
    }

    private byte[] concatenateRawContent(List<InferInput> inputs) {
        int totalSize = 0;
        for (InferInput input : inputs) {
            if (input.hasRawContent()) {
                totalSize += input.getRawContent().length;
            }
        }
        byte[] result = new byte[totalSize];
        int offset = 0;
        for (InferInput input : inputs) {
            if (input.hasRawContent()) {
                byte[] raw = input.getRawContent();
                System.arraycopy(raw, 0, result, offset, raw.length);
                offset += raw.length;
            }
        }
        return result;
    }

    // ========== Internal: Response Parsing ==========

    private InferResult parseInferResponse(HttpResponse<byte[]> response, String modelId, String modelVersion) {
        byte[] body = response.body();
        ObjectMapper mapper = helper.getObjectMapper();

        // Check for Inference-Header-Content-Length to determine binary extension format
        int jsonLength = body.length;
        String headerLengthStr = response.headers().firstValue("Inference-Header-Content-Length").orElse(null);
        if (headerLengthStr != null) {
            jsonLength = Integer.parseInt(headerLengthStr);
        }

        String jsonPart = new String(body, 0, jsonLength, StandardCharsets.UTF_8);
        byte[] binaryPart = new byte[body.length - jsonLength];
        if (binaryPart.length > 0) {
            System.arraycopy(body, jsonLength, binaryPart, 0, binaryPart.length);
        }

        try {
            JsonNode jsonResponse = mapper.readTree(jsonPart);
            String respModelName = jsonResponse.path("model_name").asText(modelId);
            String respModelVersion = jsonResponse.path("model_version").asText(modelVersion != null ? modelVersion : "");
            String requestId = jsonResponse.path("id").asText("");

            List<OutputTensorDescriptor> outputDescriptors = new ArrayList<>();
            JsonNode outputsNode = jsonResponse.path("outputs");
            int binaryOffset = 0;

            if (outputsNode.isArray()) {
                for (JsonNode outputNode : outputsNode) {
                    String name = outputNode.path("name").asText();
                    String datatype = outputNode.path("datatype").asText();
                    JsonNode shapeNode = outputNode.path("shape");
                    long[] shape = new long[shapeNode.size()];
                    for (int i = 0; i < shapeNode.size(); i++) {
                        shape[i] = shapeNode.get(i).asLong();
                    }

                    byte[] rawContent;
                    JsonNode binaryDataSizeNode = outputNode.path("parameters").path("binary_data_size");
                    if (!binaryDataSizeNode.isMissingNode()) {
                        int binaryDataSize = binaryDataSizeNode.asInt();
                        rawContent = new byte[binaryDataSize];
                        System.arraycopy(binaryPart, binaryOffset, rawContent, 0, binaryDataSize);
                        binaryOffset += binaryDataSize;
                    } else {
                        // Fallback: data is in the JSON "data" field
                        rawContent = jsonDataToRawContent(outputNode.path("data"), datatype, shape);
                    }

                    outputDescriptors.add(new OutputTensorDescriptor(name, datatype, shape, rawContent));
                }
            }

            return new InferResult(respModelName, respModelVersion, requestId, outputDescriptors);
        } catch (TritonInferException e) {
            throw e;
        } catch (Exception e) {
            throw new TritonInferException("Failed to parse inference response", e);
        }
    }

    /**
     * Parses a generate_stream SSE event where outputs are flat top-level keys.
     * Format: {"model_name":"m","model_version":"1","TEXT_OUTPUT":"token"}
     */
    private InferResult parseSseGenerateEvent(String jsonData, String modelId, String modelVersion) {
        ObjectMapper mapper = helper.getObjectMapper();
        try {
            JsonNode json = mapper.readTree(jsonData);
            String respModelName = json.path("model_name").asText(modelId);
            String respModelVersion = json.path("model_version").asText(modelVersion != null ? modelVersion : "");
            String requestId = json.path("id").asText("");

            // Reserved keys that are metadata, not output tensors
            java.util.Set<String> reservedKeys = java.util.Set.of(
                    "model_name", "model_version", "id", "parameters");

            List<OutputTensorDescriptor> outputDescriptors = new ArrayList<>();
            var fields = json.fields();
            while (fields.hasNext()) {
                var entry = fields.next();
                String key = entry.getKey();
                if (reservedKeys.contains(key)) continue;

                JsonNode value = entry.getValue();
                if (value.isTextual()) {
                    // STRING/BYTES output - serialize as length-prefixed UTF-8
                    byte[] strBytes = value.asText().getBytes(StandardCharsets.UTF_8);
                    ByteBuffer buf = ByteBuffer.allocate(4 + strBytes.length).order(ByteOrder.LITTLE_ENDIAN);
                    buf.putInt(strBytes.length);
                    buf.put(strBytes);
                    outputDescriptors.add(new OutputTensorDescriptor(key, "BYTES", new long[]{1}, buf.array()));
                } else if (value.isNumber()) {
                    if (value.isFloatingPointNumber()) {
                        ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                        buf.putFloat((float) value.asDouble());
                        outputDescriptors.add(new OutputTensorDescriptor(key, "FP32", new long[]{1}, buf.array()));
                    } else {
                        ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                        buf.putInt(value.asInt());
                        outputDescriptors.add(new OutputTensorDescriptor(key, "INT32", new long[]{1}, buf.array()));
                    }
                } else if (value.isBoolean()) {
                    byte[] raw = {(byte) (value.asBoolean() ? 1 : 0)};
                    outputDescriptors.add(new OutputTensorDescriptor(key, "BOOL", new long[]{1}, raw));
                } else if (value.isArray()) {
                    // Array output - check first element type
                    if (value.size() > 0 && value.get(0).isTextual()) {
                        String[] strings = new String[value.size()];
                        for (int i = 0; i < value.size(); i++) strings[i] = value.get(i).asText();
                        byte[] raw = serializeStringsToBytes(strings);
                        outputDescriptors.add(new OutputTensorDescriptor(key, "BYTES", new long[]{value.size()}, raw));
                    } else {
                        ByteBuffer buf = ByteBuffer.allocate(value.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
                        for (JsonNode v : value) buf.putFloat((float) v.asDouble());
                        outputDescriptors.add(new OutputTensorDescriptor(key, "FP32", new long[]{value.size()}, buf.array()));
                    }
                }
            }

            return new InferResult(respModelName, respModelVersion, requestId, outputDescriptors);
        } catch (TritonInferException e) {
            throw e;
        } catch (Exception e) {
            throw new TritonInferException("Failed to parse SSE generate event: " + jsonData, e);
        }
    }

    private byte[] serializeStringsToBytes(String[] strings) {
        int totalSize = 0;
        byte[][] encoded = new byte[strings.length][];
        for (int i = 0; i < strings.length; i++) {
            encoded[i] = strings[i].getBytes(StandardCharsets.UTF_8);
            totalSize += 4 + encoded[i].length;
        }
        ByteBuffer buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN);
        for (byte[] bytes : encoded) {
            buf.putInt(bytes.length);
            buf.put(bytes);
        }
        return buf.array();
    }

    /**
     * Converts JSON data array to raw binary content (Little-Endian).
     * Used when binary extension is not used (SSE streaming, JSON data field).
     */
    private byte[] jsonDataToRawContent(JsonNode dataNode, String datatype, long[] shape) {
        if (dataNode == null || dataNode.isMissingNode() || !dataNode.isArray()) {
            return new byte[0];
        }

        TritonDataType dtype = TritonDataType.fromString(datatype);
        return switch (dtype) {
            case BOOL -> {
                ByteBuffer buf = ByteBuffer.allocate(dataNode.size()).order(ByteOrder.LITTLE_ENDIAN);
                for (JsonNode val : dataNode) buf.put((byte) (val.asBoolean() ? 1 : 0));
                yield buf.array();
            }
            case INT8, UINT8 -> {
                byte[] arr = new byte[dataNode.size()];
                for (int i = 0; i < dataNode.size(); i++) arr[i] = (byte) dataNode.get(i).asInt();
                yield arr;
            }
            case INT16, UINT16 -> {
                ByteBuffer buf = ByteBuffer.allocate(dataNode.size() * 2).order(ByteOrder.LITTLE_ENDIAN);
                for (JsonNode val : dataNode) buf.putShort((short) val.asInt());
                yield buf.array();
            }
            case INT32, UINT32 -> {
                ByteBuffer buf = ByteBuffer.allocate(dataNode.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
                for (JsonNode val : dataNode) buf.putInt(val.asInt());
                yield buf.array();
            }
            case INT64, UINT64 -> {
                ByteBuffer buf = ByteBuffer.allocate(dataNode.size() * 8).order(ByteOrder.LITTLE_ENDIAN);
                for (JsonNode val : dataNode) buf.putLong(val.asLong());
                yield buf.array();
            }
            case FP32 -> {
                ByteBuffer buf = ByteBuffer.allocate(dataNode.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
                for (JsonNode val : dataNode) buf.putFloat((float) val.asDouble());
                yield buf.array();
            }
            case FP64 -> {
                ByteBuffer buf = ByteBuffer.allocate(dataNode.size() * 8).order(ByteOrder.LITTLE_ENDIAN);
                for (JsonNode val : dataNode) buf.putDouble(val.asDouble());
                yield buf.array();
            }
            case BYTES -> {
                // Length-prefixed UTF-8 strings
                int totalSize = 0;
                byte[][] encoded = new byte[dataNode.size()][];
                for (int i = 0; i < dataNode.size(); i++) {
                    encoded[i] = dataNode.get(i).asText().getBytes(StandardCharsets.UTF_8);
                    totalSize += 4 + encoded[i].length;
                }
                ByteBuffer buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN);
                for (byte[] bytes : encoded) {
                    buf.putInt(bytes.length);
                    buf.put(bytes);
                }
                yield buf.array();
            }
            default -> new byte[0];
        };
    }
}
