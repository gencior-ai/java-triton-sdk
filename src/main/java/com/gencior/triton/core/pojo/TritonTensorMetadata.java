package com.gencior.triton.core.pojo;

import java.util.Collections;
import java.util.List;

import inference.GrpcService;

/**
 * Encapsulates metadata information about a single tensor in a model's input or output.
 *
 * <p>This class describes the schema of a tensor including its name, data type, and shape.
 * This information is essential for correctly formatting data before sending it to Triton
 * or interpreting data returned from Triton.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code ModelMetadataResponse.TensorMetadata}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonTensorMetadata {

    private final String name;
    private final String datatype;
    private final List<Long> shape;

    private TritonTensorMetadata(String name, String datatype, List<Long> shape) {
        this.name = name;
        this.datatype = datatype;
        this.shape = shape != null ? List.copyOf(shape) : Collections.emptyList();
    }

    public static TritonTensorMetadata fromProto(GrpcService.ModelMetadataResponse.TensorMetadata proto) {
        return new TritonTensorMetadata(
            proto.getName(),
            proto.getDatatype(),
            proto.getShapeList()
        );
    }

    /**
     * Returns the name of the tensor.
     *
     * @return the tensor name
     */
    public String getName() { return name; }

    /**
     * Returns the data type of the tensor elements.
     *
     * <p>Examples: "FP32", "INT64", "BYTES", etc.
     *
     * @return the data type as a string
     */
    public String getDatatype() { return datatype; }

    /**
     * Returns an unmodifiable list representing the shape of the tensor.
     *
     * <p>For example, a 3D image might have shape [1, 224, 224]. A dimension value of -1
     * indicates a variable-sized dimension.
     *
     * <p>The returned list is immutable and cannot be modified.
     *
     * @return an immutable list of dimension sizes
     */
    public List<Long> getShape() { return shape; }

    @Override
    public String toString() {
        return String.format("Tensor[name=%s, type=%s, shape=%s]", name, datatype, shape);
    }
}
