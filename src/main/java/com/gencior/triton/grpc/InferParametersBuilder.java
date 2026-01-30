package com.gencior.triton.grpc;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import inference.GrpcService.InferParameter;

/**
 * Builder for constructing custom inference request parameters.
 *
 * <p>This class provides a fluent API for building a map of custom parameters to send with
 * inference requests to Triton. Parameters are name-value pairs that can be used to control
 * various aspects of inference execution, such as request priority, binary data output options,
 * and model-specific behavior.
 *
 * <p>Reserved parameter names (sequence_id, sequence_start, sequence_end, priority, binary_data_output)
 * are protected and cannot be set using this builder as they are managed separately.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * Map<String, InferParameter> params = new InferParametersBuilder()
 *     .add("temperature", 0.7)
 *     .add("max_tokens", 100L)
 *     .add("use_cache", true)
 *     .build();
 *
 * InferResult result = client.infer(modelId, inputs, params);
 * }</pre>
 *
 * <h2>Supported Parameter Types:</h2>
 * <ul>
 *   <li><strong>String</strong> - Text values via {@link #add(String, String)}</li>
 *   <li><strong>Long</strong> - Integer values via {@link #add(String, long)}</li>
 *   <li><strong>Boolean</strong> - Boolean flags via {@link #add(String, boolean)}</li>
 *   <li><strong>Double</strong> - Floating-point values via {@link #add(String, double)}</li>
 * </ul>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public class InferParametersBuilder {
    private final Map<String, InferParameter> parameters = new HashMap<>();

    private static final Set<String> RESERVED_KEYS = Set.of(
        "sequence_id", "sequence_start", "sequence_end", "priority", "binary_data_output"
    );

    /**
     * Adds a string-valued parameter to the inference request.
     *
     * @param key the parameter name (must not be a reserved parameter)
     * @param value the string value for this parameter
     * @return this builder instance for method chaining
     * @throws IllegalArgumentException if the key is a reserved parameter name
     */
    public InferParametersBuilder add(String key, String value) {
        checkReserved(key);
        parameters.put(key, InferParameter.newBuilder().setStringParam(value).build());
        return this;
    }

    /**
     * Adds a long integer-valued parameter to the inference request.
     *
     * @param key the parameter name (must not be a reserved parameter)
     * @param value the long integer value for this parameter
     * @return this builder instance for method chaining
     * @throws IllegalArgumentException if the key is a reserved parameter name
     */
    public InferParametersBuilder add(String key, long value) {
        checkReserved(key);
        parameters.put(key, InferParameter.newBuilder().setInt64Param(value).build());
        return this;
    }

    /**
     * Adds a boolean-valued parameter to the inference request.
     *
     * @param key the parameter name (must not be a reserved parameter)
     * @param value the boolean value for this parameter
     * @return this builder instance for method chaining
     * @throws IllegalArgumentException if the key is a reserved parameter name
     */
    public InferParametersBuilder add(String key, boolean value) {
        checkReserved(key);
        parameters.put(key, InferParameter.newBuilder().setBoolParam(value).build());
        return this;
    }

    /**
     * Adds a double-valued parameter to the inference request.
     *
     * @param key the parameter name (must not be a reserved parameter)
     * @param value the double (floating-point) value for this parameter
     * @return this builder instance for method chaining
     * @throws IllegalArgumentException if the key is a reserved parameter name
     */
    public InferParametersBuilder add(String key, double value) {
        checkReserved(key);
        parameters.put(key, InferParameter.newBuilder().setDoubleParam(value).build());
        return this;
    }

    /**
     * Validates that a parameter key is not reserved.
     *
     * <p>Reserved parameters are managed separately by the SDK and cannot be set via this builder.
     * Reserved parameter names are: sequence_id, sequence_start, sequence_end, priority, binary_data_output.
     *
     * @param key the parameter name to validate
     * @throws IllegalArgumentException if the key is a reserved parameter name
     */
    private void checkReserved(String key) {
        if (RESERVED_KEYS.contains(key)) {
            throw new IllegalArgumentException("Parameter \"" + key + "\" is a reserved parameter and cannot be specified.");
        }
    }

    /**
     * Adds a parameter to the builder without reserved key validation.
     *
     * <p>This method is for internal use only and bypasses the reserved key check.
     * Used internally to add reserved parameters that are managed by the SDK.
     *
     * @param key the parameter name
     * @param parameter the InferParameter protobuf message
     */
    protected void addInternal(String key, InferParameter parameter) {
        parameters.put(key, parameter);
    }

    /**
     * Builds and returns the accumulated parameters map.
     *
     * @return an unmodifiable map of parameter names to InferParameter values
     */
    public Map<String, InferParameter> build() {
        return parameters;
    }
}
