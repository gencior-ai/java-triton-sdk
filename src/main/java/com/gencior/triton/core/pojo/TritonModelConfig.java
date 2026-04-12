package com.gencior.triton.core.pojo;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;

import inference.ModelConfigOuterClass;

/**
 * Encapsulates configuration information for a Triton model.
 *
 * <p>This class represents the runtime configuration of a deployed model including its platform,
 * backend, runtime environment, batching capabilities, and associated model files. This
 * configuration is read from Triton's model configuration files and defines how the model is
 * executed and what data formats it expects.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code ModelConfig}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonModelConfig {

    private final String name;
    private final String platform;
    private final String backend;
    private final String runtime;
    private final int maxBatchSize;
    private final String defaultModelFilename;
    private final Map<String, String> ccModelFilenames;
    private final Map<String, String> metricTags;

    private TritonModelConfig(String name, String platform, String backend, String runtime, 
                             int maxBatchSize, String defaultModelFilename, 
                             Map<String, String> ccModelFilenames, Map<String, String> metricTags) {
        this.name = name;
        this.platform = platform;
        this.backend = backend;
        this.runtime = runtime;
        this.maxBatchSize = maxBatchSize;
        this.defaultModelFilename = defaultModelFilename;
        this.ccModelFilenames = ccModelFilenames != null ? Map.copyOf(ccModelFilenames) : Collections.emptyMap();
        this.metricTags = metricTags != null ? Map.copyOf(metricTags) : Collections.emptyMap();
    }

    public static TritonModelConfig fromProto(ModelConfigOuterClass.ModelConfig proto) {
        return new TritonModelConfig(
            proto.getName(),
            proto.getPlatform(),
            proto.getBackend(),
            proto.getRuntime(),
            proto.getMaxBatchSize(),
            proto.getDefaultModelFilename(),
            proto.getCcModelFilenamesMap(),
            proto.getMetricTagsMap()
        );
    }

    /**
     * Creates a TritonModelConfig from a JSON response.
     *
     * @param json the JSON node containing model configuration fields such as
     *             {@code name}, {@code backend}, {@code platform}, {@code max_batch_size}
     * @return a new TritonModelConfig instance
     */
    public static TritonModelConfig fromJson(JsonNode json) {
        return new TritonModelConfig(
                json.path("name").asText(""),
                json.path("platform").asText(""),
                json.path("backend").asText(""),
                json.path("runtime").asText(""),
                json.path("max_batch_size").asInt(0),
                json.path("default_model_filename").asText(""),
                parseStringMap(json.path("cc_model_filenames")),
                parseStringMap(json.path("metric_tags"))
        );
    }

    private static Map<String, String> parseStringMap(JsonNode node) {
        Map<String, String> map = new HashMap<>();
        if (node.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> entry = fields.next();
                map.put(entry.getKey(), entry.getValue().asText(""));
            }
        }
        return map;
    }

    /** Returns the name of the model. */
    public String getName() { return name; }

    /** Returns the platform type (e.g., "tensorflow_savedmodel", "tensorrt_plan"). */
    public String getPlatform() { return platform; }

    /** Returns the backend implementation name. */
    public String getBackend() { return backend; }

    /** Returns the runtime environment name. */
    public String getRuntime() { return runtime; }

    /** Returns the maximum batch size supported by the model, or 0 if batching is disabled. */
    public int getMaxBatchSize() { return maxBatchSize; }

    /** Returns the default model filename. */
    public String getDefaultModelFilename() { return defaultModelFilename; }

    /** Returns an unmodifiable map of compute capability to model filename mappings. */
    public Map<String, String> getCcModelFilenames() { return ccModelFilenames; }

    /** Returns an unmodifiable map of custom metric tags for this model. */
    public Map<String, String> getMetricTags() { return metricTags; }

    @Override
    public String toString() {
        return String.format("TritonModelConfig[name=%s, backend=%s, maxBatchSize=%d]", 
            name, (backend.isEmpty() ? platform : backend), maxBatchSize);
    }
}