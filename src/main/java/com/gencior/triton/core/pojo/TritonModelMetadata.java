package com.gencior.triton.core.pojo;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import inference.GrpcService;

/**
 * Encapsulates metadata information about a Triton model.
 *
 * <p>This class represents model-level metadata including the model name, available versions,
 * platform type, and the schema for input and output tensors. This information describes the
 * expected interface for communicating with a specific model deployed on Triton.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code ModelMetadataResponse}.
 *
 * <h2>Typical Usage:</h2>
 * <pre>{@code
 * TritonModelMetadata metadata = client.modelMetadata("inception_v3");
 * System.out.println("Model: " + metadata.getName());
 * System.out.println("Platform: " + metadata.getPlatform());
 * System.out.println("Inputs: " + metadata.getInputs());
 * System.out.println("Outputs: " + metadata.getOutputs());
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonModelMetadata {

    private final String name;
    private final List<String> versions;
    private final String platform;
    private final List<TritonTensorMetadata> inputs;
    private final List<TritonTensorMetadata> outputs;

    private TritonModelMetadata(String name, List<String> versions, String platform, 
                               List<TritonTensorMetadata> inputs, List<TritonTensorMetadata> outputs) {
        this.name = name;
        this.versions = versions != null ? List.copyOf(versions) : Collections.emptyList();
        this.platform = platform;
        this.inputs = inputs != null ? List.copyOf(inputs) : Collections.emptyList();
        this.outputs = outputs != null ? List.copyOf(outputs) : Collections.emptyList();
    }

    public static TritonModelMetadata fromProto(GrpcService.ModelMetadataResponse response) {
        List<TritonTensorMetadata> inputs = response.getInputsList().stream()
                .map(TritonTensorMetadata::fromProto)
                .collect(Collectors.toList());

        List<TritonTensorMetadata> outputs = response.getOutputsList().stream()
                .map(TritonTensorMetadata::fromProto)
                .collect(Collectors.toList());

        return new TritonModelMetadata(
            response.getName(),
            response.getVersionsList(),
            response.getPlatform(),
            inputs,
            outputs
        );
    }

    /**
     * Returns the name of the model.
     *
     * @return the model name
     */
    public String getName() { return name; }

    /**
     * Returns an unmodifiable list of available model versions.
     *
     * <p>The returned list is immutable and cannot be modified.
     *
     * @return an immutable list of version strings
     */
    public List<String> getVersions() { return versions; }

    /**
     * Returns the platform type of the model (e.g., "tensorflow_savedmodel", "tensorrt_plan").
     *
     * @return the model platform type
     */
    public String getPlatform() { return platform; }

    /**
     * Returns an unmodifiable list of input tensor metadata.
     *
     * <p>Each element describes the expected schema of an input tensor including its name,
     * shape, and data type.
     *
     * <p>The returned list is immutable and cannot be modified.
     *
     * @return an immutable list of input tensor metadata
     */
    public List<TritonTensorMetadata> getInputs() { return inputs; }

    /**
     * Returns an unmodifiable list of output tensor metadata.
     *
     * <p>Each element describes the schema of an output tensor including its name,
     * shape, and data type.
     *
     * <p>The returned list is immutable and cannot be modified.
     *
     * @return an immutable list of output tensor metadata
     */
    public List<TritonTensorMetadata> getOutputs() { return outputs; }

    @Override
    public String toString() {
        return String.format("ModelMetadata[name=%s, platform=%s, inputs=%d, outputs=%d]", 
            name, platform, inputs.size(), outputs.size());
    }
}