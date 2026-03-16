package com.gencior.triton.core;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Specifies an output tensor to be returned from an inference request.
 *
 * <p>By default, Triton returns all outputs defined in a model's configuration.
 * Use this class to request only specific outputs, reducing network bandwidth
 * and memory usage — especially useful for models with many output tensors
 * where only a subset is needed.
 *
 * <h2>Usage — Simple (name only):</h2>
 * <pre>{@code
 * InferResult result = client.infer("bert", inputs,
 *     List.of(InferRequestedOutput.of("embeddings")));
 * // Only "embeddings" is returned, other outputs are skipped
 * }</pre>
 *
 * <h2>Usage — With parameters:</h2>
 * <pre>{@code
 * InferRequestedOutput output = new InferRequestedOutput.Builder("classification")
 *     .addParameter("classification", 3L) // top-3 classes
 *     .build();
 * InferResult result = client.infer("classifier", inputs, List.of(output));
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class InferRequestedOutput {

    private final String name;
    private final Map<String, Object> parameters;

    private InferRequestedOutput(String name, Map<String, Object> parameters) {
        this.name = Objects.requireNonNull(name, "output name must not be null");
        this.parameters = Collections.unmodifiableMap(new HashMap<>(parameters));
    }

    /**
     * Creates a requested output with just a name and no additional parameters.
     *
     * @param name the output tensor name (must match the model's output name)
     * @return a new InferRequestedOutput
     */
    public static InferRequestedOutput of(String name) {
        return new InferRequestedOutput(name, Map.of());
    }

    /**
     * Returns the output tensor name.
     *
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the optional parameters for this output.
     *
     * @return an unmodifiable map of parameters (may be empty)
     */
    public Map<String, Object> getParameters() {
        return parameters;
    }

    /**
     * Returns whether this output has any parameters set.
     *
     * @return {@code true} if parameters are present
     */
    public boolean hasParameters() {
        return !parameters.isEmpty();
    }

    /**
     * Builder for constructing {@link InferRequestedOutput} instances
     * with optional parameters.
     *
     * @since 1.0.0
     */
    public static class Builder {
        private final String name;
        private final Map<String, Object> parameters = new HashMap<>();

        /**
         * Creates a builder for the given output tensor name.
         *
         * @param name the output tensor name
         */
        public Builder(String name) {
            this.name = Objects.requireNonNull(name, "output name must not be null");
        }

        public Builder addParameter(String key, String value) {
            parameters.put(key, value);
            return this;
        }

        public Builder addParameter(String key, long value) {
            parameters.put(key, value);
            return this;
        }

        public Builder addParameter(String key, boolean value) {
            parameters.put(key, value);
            return this;
        }

        public Builder addParameter(String key, double value) {
            parameters.put(key, value);
            return this;
        }

        public InferRequestedOutput build() {
            return new InferRequestedOutput(name, parameters);
        }
    }
}
