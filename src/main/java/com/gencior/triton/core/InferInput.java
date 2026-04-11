package com.gencior.triton.core;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.gencior.triton.exceptions.TritonDataNotFoundException;
import com.gencior.triton.exceptions.TritonDataTypeException;
import com.gencior.triton.exceptions.TritonShapeMismatchException;

import inference.GrpcService;

/**
 * An object of {@code InferInput} class is used to describe an input tensor for
 * an inference request.
 * * <p>
 * This class is designed to be aligned with the Python
 * {@:tritonclient.grpc.InferInput} implementation. It provides a high-level API
 * to handle data serialization into the raw byte format required by Triton,
 * supporting various data types and shared memory regions.</p>
 *
 * <p>
 * Key features include:</p>
 * <ul>
 * <li>Automatic Little-Endian serialization for numeric types.</li>
 * <li>Length-prefixed serialization for {@code BYTES} (String) data types.</li>
 * <li>Data size validation against the provided tensor shape.</li>
 * <li>Support for System and CUDA Shared Memory parameters.</li>
 * </ul>
 *
 * @author sachachoumiloff
 */
public class InferInput {

    private final String name;
    private final String datatype;
    private final List<Long> shape;
    private final Map<String, Object> parameters;
    private byte[] rawContent;

    /**
     * Creates an {@code InferInput} instance.
     *
     * @param name The name of the input.
     * @param shape The shape of the associated input.
     * @param datatype The {@link TritonDataType} of the associated input.
     */
    public InferInput(String name, long[] shape, TritonDataType datatype) {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(shape, "shape must not be null");
        Objects.requireNonNull(datatype, "datatype must not be null");
        this.name = name;
        this.datatype = datatype.getTritonName();
        this.shape = new ArrayList<>();
        for (long dim : shape) {
            this.shape.add(dim);
        }
        this.parameters = new HashMap<>();
        this.rawContent = null;
    }

    /**
     * @return The name of the input associated with this Input Tensor.
     */
    public String getName() {
        return name;
    }

    /**
     * @return The datatype of the input associated with this object.
     */
    public TritonDataType getDatatype() {
        return TritonDataType.fromString(datatype);
    }

    /**
     * @return The Triton datatype name as a raw string (e.g. "FP32", "INT32").
     */
    public String getDatatypeString() {
        return datatype;
    }

    /**
     * @return The current shape of the input as an array of longs.
     */
    public long[] getShape() {
        return shape.stream().mapToLong(Long::longValue).toArray();
    }

    /**
     * Updates the shape of the input. Useful for models with dynamic input
     * shapes.
     *
     * * @param shape The new shape for the associated input.
     * @return This {@code InferInput} instance for method chaining.
     */
    public InferInput setShape(long[] shape) {
        this.shape.clear();
        for (long dim : shape) {
            this.shape.add(dim);
        }
        return this;
    }

    /**
     * @return An unmodifiable copy of the parameters map.
     */
    public Map<String, Object> getParameters() {
        return Collections.unmodifiableMap(parameters);
    }

    /**
     * Sets the tensor data from a float array. Validates that the input matches
     * the expected "FP32" datatype and size.
     *
     * * @param data The float array to be used as input.
     * @return This {@code InferInput} instance for method chaining.
     * @throws TritonDataTypeException If data size mismatch.
     * @throws TritonShapeMismatchException If data size mismatch.
     */
    public InferInput setData(float[] data) {
        validateDataSize(data.length);
        validateDatatype(TritonDataType.FP32);
        ByteBuffer buffer = ByteBuffer.allocate(data.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) {
            buffer.putFloat(v);
        }
        this.rawContent = buffer.array();
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Sets the tensor data from a double array (FP64).
     *
     * * @param data The double array to be used as input.
     * @return This {@code InferInput} instance for method chaining.
     */
    public InferInput setData(double[] data) {
        validateDataSize(data.length);
        validateDatatype(TritonDataType.FP64);
        ByteBuffer buffer = ByteBuffer.allocate(data.length * Double.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : data) {
            buffer.putDouble(v);
        }
        this.rawContent = buffer.array();
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Sets the tensor data from an int array. Supports INT8, INT16, INT32,
     * UINT8, UINT16, and UINT32 datatypes. Serialization byte size is
     * determined by the declared datatype.
     *
     * @param data The int array to be used as input.
     * @return This {@code InferInput} instance for method chaining.
     * @throws TritonDataTypeException If the tensor datatype is incompatible.
     * @throws TritonShapeMismatchException If data size does not match the shape.
     */
    public InferInput setData(int[] data) {
        validateDataSize(data.length);
        validateDatatype(TritonDataType.INT8, TritonDataType.INT16, TritonDataType.INT32,
                TritonDataType.UINT8, TritonDataType.UINT16, TritonDataType.UINT32);
        TritonDataType dtype = getDatatype();
        int elementSize = switch (dtype) {
            case INT8, UINT8 -> Byte.BYTES;
            case INT16, UINT16 -> Short.BYTES;
            default -> Integer.BYTES;
        };
        ByteBuffer buffer = ByteBuffer.allocate(data.length * elementSize).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : data) {
            switch (dtype) {
                case INT8, UINT8 -> buffer.put((byte) v);
                case INT16, UINT16 -> buffer.putShort((short) v);
                default -> buffer.putInt(v);
            }
        }
        this.rawContent = buffer.array();
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Sets the tensor data from a short array. Supports INT16, UINT16
     * datatypes.
     *
     * @param data The short array to be used as input.
     * @return This {@code InferInput} instance for method chaining.
     * @throws TritonDataTypeException If the tensor datatype is incompatible.
     * @throws TritonShapeMismatchException If data size does not match the shape.
     */
    public InferInput setData(short[] data) {
        validateDataSize(data.length);
        validateDatatype(TritonDataType.INT16, TritonDataType.UINT16);
        ByteBuffer buffer = ByteBuffer.allocate(data.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (short v : data) {
            buffer.putShort(v);
        }
        this.rawContent = buffer.array();
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Sets the tensor data from a long array. Supports INT64 and UINT64
     * datatypes.
     *
     * @param data The long array to be used as input.
     * @return This {@code InferInput} instance for method chaining.
     * @throws TritonDataTypeException If the tensor datatype is incompatible.
     * @throws TritonShapeMismatchException If data size does not match the shape.
     */
    public InferInput setData(long[] data) {
        validateDataSize(data.length);
        validateDatatype(TritonDataType.INT64, TritonDataType.UINT64);
        ByteBuffer buffer = ByteBuffer.allocate(data.length * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (long v : data) {
            buffer.putLong(v);
        }
        this.rawContent = buffer.array();
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Sets the tensor data from a boolean array. Internally converts booleans
     * to 1-byte integers (1 for true, 0 for false).
     *
     * * @param data The boolean array to be used as input.
     * @return This {@code InferInput} instance for method chaining.
     */
    public InferInput setData(boolean[] data) {
        validateDataSize(data.length);
        validateDatatype(TritonDataType.BOOL);
        ByteBuffer buffer = ByteBuffer.allocate(data.length).order(ByteOrder.LITTLE_ENDIAN);
        for (boolean v : data) {
            buffer.put((byte) (v ? 1 : 0));
        }
        this.rawContent = buffer.array();
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Sets the tensor data from raw bytes. Use this if you have already
     * serialized the data externally.
     *
     * * @param data The raw byte array.
     * @return This {@code InferInput} instance for method chaining.
     */
    public InferInput setData(byte[] data) {
        this.rawContent = data;
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Sets the tensor data from a String array (BYTES datatype). Each string is
     * serialized with a 4-byte Little-Endian length prefix followed by the
     * UTF-8 encoded string bytes.
     *
     * * @param data The string array to be used as input.
     * @return This {@code InferInput} instance for method chaining.
     */
    public InferInput setData(String[] data) {
        validateDataSize(data.length);
        validateDatatype(TritonDataType.BYTES);
        this.rawContent = serializeStrings(data);
        clearSharedMemoryParams();
        return this;
    }

    /**
     * Returns the underlying Protobuf message built on-the-fly from the
     * internal fields. Provided for backward compatibility with the gRPC
     * transport layer.
     *
     * @return The {@code InferInputTensor} message.
     */
    public GrpcService.ModelInferRequest.InferInputTensor getTensor() {
        GrpcService.ModelInferRequest.InferInputTensor.Builder builder =
                GrpcService.ModelInferRequest.InferInputTensor.newBuilder()
                        .setName(name)
                        .setDatatype(datatype);
        for (long dim : shape) {
            builder.addShape(dim);
        }
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
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
            }
            builder.putParameters(entry.getKey(), paramBuilder.build());
        }
        return builder.build();
    }

    /**
     * Returns the serialized binary content of the tensor.
     *
     * * @return A byte array containing the tensor data.
     */
    public byte[] getRawContent() {
        return rawContent;
    }

    /**
     * @return {@code true} if raw content has been set and is not empty.
     */
    public boolean hasRawContent() {
        return rawContent != null && rawContent.length > 0;
    }

    private void validateDataSize(int dataLength) {
        long expectedSize = calculateExpectedSize();
        if (expectedSize >= 0 && dataLength != expectedSize) {
            throw new TritonShapeMismatchException(
                    getShape(),
                    dataLength,
                    expectedSize
            );
        }
    }

    private void validateDatatype(TritonDataType... expectedTypes) {
        boolean match = Arrays.stream(expectedTypes)
                .anyMatch(type -> type.getTritonName().equals(this.datatype));

        if (!match) {
            throw new TritonDataTypeException(String.format(
                    "Datatype mismatch: expected one of %s but tensor has datatype %s",
                    Arrays.toString(expectedTypes), this.datatype));
        }
    }

    private long calculateExpectedSize() {
        long[] shape = getShape();
        if (shape.length == 0) {
            return 0;
        }
        long size = 1;
        for (long dim : shape) {
            size *= dim;
        }
        return size;
    }

    private void clearSharedMemoryParams() {
        parameters.remove("shared_memory_region");
        parameters.remove("shared_memory_byte_size");
        parameters.remove("shared_memory_offset");
    }

    private byte[] serializeStrings(String[] strings) {
        int totalSize = 0;
        byte[][] encoded = new byte[strings.length][];
        for (int i = 0; i < strings.length; i++) {
            encoded[i] = strings[i].getBytes(StandardCharsets.UTF_8);
            totalSize += 4 + encoded[i].length; // 4 bytes for length prefix
        }

        ByteBuffer buffer = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN);
        for (byte[] bytes : encoded) {
            buffer.putInt(bytes.length);
            buffer.put(bytes);
        }
        return buffer.array();
    }

    /**
     * Reconstructs a float array from the internal raw content.
     *
     * @return A float array representation of the data.
     * @throws InferDataException If no content is available.
     */
    public float[] getDataAsFloatArray() {
        checkRawContentExists();
        ByteBuffer buffer = ByteBuffer.wrap(rawContent).order(ByteOrder.LITTLE_ENDIAN);
        float[] result = new float[rawContent.length / Float.BYTES];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.getFloat();
        }
        return result;
    }

    /**
     * Reconstructs a double array from the internal raw content.
     *
     * @return A double array representation of the data.
     */
    public double[] getDataAsDoubleArray() {
        checkRawContentExists();
        ByteBuffer buffer = ByteBuffer.wrap(rawContent).order(ByteOrder.LITTLE_ENDIAN);
        double[] result = new double[rawContent.length / Double.BYTES];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.getDouble();
        }
        return result;
    }

    /**
     * Reconstructs an int array from the internal raw content.
     *
     * @return An int array representation of the data.
     */
    public int[] getDataAsIntArray() {
        checkRawContentExists();
        ByteBuffer buffer = ByteBuffer.wrap(rawContent).order(ByteOrder.LITTLE_ENDIAN);
        int[] result = new int[rawContent.length / Integer.BYTES];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.getInt();
        }
        return result;
    }

    /**
     * Reconstructs a long array from the internal raw content.
     *
     * @return A long array representation of the data.
     */
    public long[] getDataAsLongArray() {
        checkRawContentExists();
        ByteBuffer buffer = ByteBuffer.wrap(rawContent).order(ByteOrder.LITTLE_ENDIAN);
        long[] result = new long[rawContent.length / Long.BYTES];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.getLong();
        }
        return result;
    }

    /**
     * Reconstructs a boolean array from the internal raw content.
     *
     * @return A boolean array representation of the data.
     */
    public boolean[] getDataAsBooleanArray() {
        checkRawContentExists();
        boolean[] result = new boolean[rawContent.length];
        for (int i = 0; i < rawContent.length; i++) {
            result[i] = rawContent[i] != 0;
        }
        return result;
    }

    /**
     * Reconstructs a String array from the internal raw content (BYTES format).
     *
     * @return A String array representation of the data.
     */
    public String[] getDataAsStringArray() {
        checkRawContentExists();
        ByteBuffer buffer = ByteBuffer.wrap(rawContent).order(ByteOrder.LITTLE_ENDIAN);
        ArrayList<String> strings = new ArrayList<>();
        while (buffer.hasRemaining()) {
            int length = buffer.getInt();
            byte[] bytes = new byte[length];
            buffer.get(bytes);
            strings.add(new String(bytes, StandardCharsets.UTF_8));
        }
        return strings.toArray(String[]::new);
    }

    private void checkRawContentExists() {
        if (rawContent == null || rawContent.length == 0) {
            throw new TritonDataNotFoundException(this.getName());
        }
    }
}
