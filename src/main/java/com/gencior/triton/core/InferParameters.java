package com.gencior.triton.core;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Represents a set of custom parameters for an inference request.
 *
 * <p>This class provides a protocol-agnostic abstraction over inference
 * parameters, decoupling the public API from the underlying transport
 * (gRPC, HTTP). Values can be of type {@code String}, {@code Long},
 * {@code Boolean}, or {@code Double}.
 *
 * <p>Use the {@link Builder} to construct instances:
 * <pre>{@code
 * InferParameters params = new InferParameters.Builder()
 *     .add("temperature", 0.7)
 *     .add("max_tokens", 100L)
 *     .add("use_cache", true)
 *     .build();
 *
 * InferResult result = client.infer(modelId, inputs, params);
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class InferParameters {

    private final Map<String, Object> parameters;

    private InferParameters(Map<String, Object> parameters) {
        this.parameters = Collections.unmodifiableMap(new HashMap<>(parameters));
    }

    /**
     * Returns the parameters as an unmodifiable map.
     *
     * @return the parameters map
     */
    public Map<String, Object> asMap() {
        return parameters;
    }

    private static final Set<String> RESERVED_KEYS = Set.of(
            "sequence_id", "sequence_start", "sequence_end", "priority", "binary_data_output"
    );

    /**
     * Builder for constructing {@link InferParameters} instances.
     *
     * <p>Provides a fluent API for building custom inference parameters.
     * Reserved parameter names (sequence_id, sequence_start, sequence_end,
     * priority, binary_data_output) are protected and cannot be set as they
     * are managed by the SDK.
     *
     * @since 1.0.0
     */
    public static class Builder {
        private final Map<String, Object> parameters = new HashMap<>();

        /**
         * Adds a string-valued parameter.
         *
         * @param key the parameter name
         * @param value the string value
         * @return this builder instance for method chaining
         * @throws IllegalArgumentException if the key is reserved
         */
        public Builder add(String key, String value) {
            checkReserved(key);
            parameters.put(key, value);
            return this;
        }

        /**
         * Adds a long integer-valued parameter.
         *
         * @param key the parameter name
         * @param value the long value
         * @return this builder instance for method chaining
         * @throws IllegalArgumentException if the key is reserved
         */
        public Builder add(String key, long value) {
            checkReserved(key);
            parameters.put(key, value);
            return this;
        }

        /**
         * Adds a boolean-valued parameter.
         *
         * @param key the parameter name
         * @param value the boolean value
         * @return this builder instance for method chaining
         * @throws IllegalArgumentException if the key is reserved
         */
        public Builder add(String key, boolean value) {
            checkReserved(key);
            parameters.put(key, value);
            return this;
        }

        /**
         * Adds a double-valued parameter.
         *
         * @param key the parameter name
         * @param value the double value
         * @return this builder instance for method chaining
         * @throws IllegalArgumentException if the key is reserved
         */
        public Builder add(String key, double value) {
            checkReserved(key);
            parameters.put(key, value);
            return this;
        }

        private void checkReserved(String key) {
            if (RESERVED_KEYS.contains(key)) {
                throw new IllegalArgumentException(
                        "Parameter \"" + key + "\" is a reserved parameter and cannot be specified.");
            }
        }

        /**
         * Builds and returns a new {@link InferParameters} instance.
         *
         * @return the constructed InferParameters
         */
        public InferParameters build() {
            return new InferParameters(parameters);
        }
    }
}
