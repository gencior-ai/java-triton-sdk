package com.gencior.triton.core;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import com.gencior.triton.exceptions.TritonDataNotFoundException;
import com.gencior.triton.exceptions.TritonDataTypeException;
import com.gencior.triton.exceptions.TritonInferException;
import com.google.protobuf.ByteString;

import inference.GrpcService.ModelInferResponse;
import inference.GrpcService.ModelInferResponse.InferOutputTensor;

/**
 * An object of InferResult holds the response of an inference request and
 * provides methods to retrieve inference results.
 *
 * @author sachachoumiloff
 */
public class InferResult {

    private final ModelInferResponse result;

    /**
     * Constructs an InferResult instance that wraps a ModelInferResponse from
     * the Triton server.
     * <p>
     * This constructor takes the protobuf response message returned by the
     * Triton Inference Server and provides convenient methods to access and
     * deserialize the inference output tensors.
     * </p>
     *
     * @param result the ModelInferResponse returned by the Triton Inference
     * Server, must not be null
     * @throws NullPointerException if result is null
     */
    public InferResult(ModelInferResponse result) {
        this.result = result;
    }

    public String getRequestId() {
        return result.getId();
    }

    public String getModelName() {
        return result.getModelName();
    }

    public String getModelVersion() {
        return result.getModelVersion();
    }

    public List<String> getOutputNames() {
        return result.getOutputsList().stream().map(InferOutputTensor::getName).collect(Collectors.toList());
    }

    public float[] asFloatArray(String name) {
        return (float[]) getOutputAsArray(name);
    }

    public double[] asDoubleArray(String name) {
        return (double[]) getOutputAsArray(name);
    }

    public int[] asIntArray(String name) {
        return (int[]) getOutputAsArray(name);
    }

    public long[] asLongArray(String name) {
        return (long[]) getOutputAsArray(name);
    }

    public String[] asStringArray(String name) {
        return (String[]) getOutputAsArray(name);
    }

    /**
     * Retrieves the output tensor data as a native Java array.
     * <p>
     * This method deserializes the inference output tensor identified by the
     * given name into the appropriate Java primitive array type based on the
     * Triton datatype. The method first attempts to deserialize from raw binary
     * content for efficiency; if not available, it falls back to the contents
     * field.
     * </p>
     * <p>
     * The returned array type depends on the tensor's datatype:
     * <ul>
     * <li>BOOL → {@code boolean[]}</li>
     * <li>INT8 → {@code byte[]}</li>
     * <li>INT16 → {@code short[]}</li>
     * <li>INT32 → {@code int[]}</li>
     * <li>INT64 → {@code long[]}</li>
     * <li>UINT8 → {@code short[]}</li>
     * <li>UINT16 → {@code int[]}</li>
     * <li>UINT32 → {@code long[]}</li>
     * <li>UINT64 → {@code long[]}</li>
     * <li>FP32 → {@code float[]}</li>
     * <li>FP64 → {@code double[]}</li>
     * <li>FP16, BF16 → {@code float[]}</li>
     * <li>BYTES → {@code String[]}</li>
     * </ul>
     * </p>
     *
     * @param name the name of the output tensor to retrieve; must match the
     * tensor name returned by the model
     * @return a native array containing the deserialized tensor data
     * @throws TritonDataNotFoundException if no output tensor with the
     * specified name is found in the response
     * @throws TritonDataTypeException if the tensor's datatype is not supported
     * or invalid
     * @throws TritonInferException if the tensor is found but contains no data
     */
    public Object getOutputAsArray(String name) {
        int index = 0;
        for (InferOutputTensor output : result.getOutputsList()) {
            if (output.getName().equals(name)) {
                TritonDataType datatype;
                try {
                    datatype = TritonDataType.fromString(output.getDatatype());
                } catch (Exception e) {
                    throw new TritonDataTypeException("Unsupported or invalid datatype: " + output.getDatatype());
                }
                if (index < result.getRawOutputContentsList().size()) {
                    ByteString rawContent = result.getRawOutputContents(index);
                    return deserializeRawContent(rawContent, datatype);
                }
                if (output.hasContents()) {
                    return deserializeContents(output, datatype);
                }

                throw new TritonInferException("Tensor '" + name + "' found but contains no data.");
            }
            index++;
        }
        throw new TritonDataNotFoundException("Output tensor '" + name + "' was not found in the ModelInferResponse.");
    }

    /**
     * Retrieves the InferOutputTensor protobuf message for a specific output by
     * name.
     * <p>
     * This method provides direct access to the raw protobuf message
     * representing the output tensor. It allows fine-grained control over the
     * tensor data and metadata, including access to the tensor shape, datatype,
     * and both raw and structured content representations.
     * </p>
     * <p>
     * For most use cases, {@link #getOutputAsArray(String)} is more convenient
     * as it automatically deserializes the tensor into a native Java array. Use
     * this method when you need direct access to the protobuf message or need
     * to manually handle the tensor data.
     * </p>
     *
     * @param name the name of the output tensor to retrieve
     * @return the InferOutputTensor protobuf message with the specified name,
     * or {@code null} if no tensor with that name exists in the response
     */
    public InferOutputTensor getOutput(String name) {
        for (InferOutputTensor output : result.getOutputsList()) {
            if (output.getName().equals(name)) {
                return output;
            }
        }
        return null;
    }

    /**
     * Retrieves the complete ModelInferResponse protobuf message.
     * <p>
     * This method provides access to the entire response message from the
     * Triton Inference Server, including all output tensors, model metadata,
     * and any additional response information. This is useful when you need to
     * access multiple outputs or inspect the complete response structure.
     * </p>
     *
     * @return the underlying ModelInferResponse protobuf message; never null
     */
    public ModelInferResponse getResponse() {
        return result;
    }

    /**
     * Deserialize raw content bytes to appropriate Java array type based on
     * datatype.
     *
     * @param rawContent The raw bytes to deserialize
     * @param datatype The Triton datatype enum
     * @return The deserialized array
     * @throws TritonDataTypeException if datatype is not supported or buffer
     * size is invalid
     * @throws TritonInferException if buffer contains malformed data
     */
    private Object deserializeRawContent(ByteString rawContent, TritonDataType datatype) {
        ByteBuffer buffer = rawContent.asReadOnlyByteBuffer().order(java.nio.ByteOrder.LITTLE_ENDIAN);
        switch (datatype) {
            case BOOL -> {
                validateBufferSize(buffer, 1, "BOOL");
                return deserializeBoolArray(buffer);
            }
            case INT8 -> {
                return deserializeByteArray(buffer);
            }
            case INT16 -> {
                validateBufferSize(buffer, 2, "INT16");
                return deserializeShortArray(buffer);
            }
            case INT32 -> {
                validateBufferSize(buffer, 4, "INT32");
                return deserializeIntArray(buffer);
            }
            case INT64 -> {
                validateBufferSize(buffer, 8, "INT64");
                return deserializeLongArray(buffer);
            }
            case UINT8 -> {
                return deserializeUnsignedByteArray(buffer);
            }
            case UINT16 -> {
                validateBufferSize(buffer, 2, "UINT16");
                return deserializeUnsignedShortArray(buffer);
            }
            case UINT32 -> {
                validateBufferSize(buffer, 4, "UINT32");
                return deserializeUnsignedIntArray(buffer);
            }
            case UINT64 -> {
                validateBufferSize(buffer, 8, "UINT64");
                return deserializeUnsignedLongArray(buffer);
            }
            case FP32 -> {
                validateBufferSize(buffer, 4, "FP32");
                return deserializeFloatArray(buffer);
            }
            case FP64 -> {
                validateBufferSize(buffer, 8, "FP64");
                return deserializeDoubleArray(buffer);
            }
            case FP16 -> {
                validateBufferSize(buffer, 2, "FP16");
                return deserializeFP16Array(buffer);
            }
            case BF16 -> {
                validateBufferSize(buffer, 2, "BF16");
                return deserializeBF16Array(buffer);
            }
            case BYTES -> {
                return deserializeStringArray(buffer);
            }
            default ->
                throw new TritonDataTypeException("Unsupported datatype: " + datatype);
        }
    }

    /**
     * Validates that the buffer size is a multiple of the expected element
     * size.
     *
     * @param buffer The ByteBuffer to validate
     * @param elementSize The expected size in bytes for each element
     * @param datatype The datatype name (for error messages)
     * @throws TritonDataTypeException if buffer size is not a multiple of
     * elementSize
     */
    private void validateBufferSize(ByteBuffer buffer, int elementSize, String datatype) {
        int remaining = buffer.remaining();
        if (remaining % elementSize != 0) {
            throw new TritonDataTypeException(
                    String.format("Invalid buffer size for %s: expected multiple of %d bytes, but got %d bytes",
                            datatype, elementSize, remaining)
            );
        }
    }

    private Object deserializeContents(InferOutputTensor output, TritonDataType datatype) {
        var contents = output.getContents();

        return switch (datatype) {
            case BOOL ->
                contents.getBoolContentsList().stream().toArray(Boolean[]::new);
            case INT8, INT16, INT32 ->
                contents.getIntContentsList().stream().mapToInt(Integer::intValue).toArray();
            case INT64 ->
                contents.getInt64ContentsList().stream().mapToLong(Long::longValue).toArray();
            case UINT8, UINT16, UINT32 ->
                contents.getUintContentsList().stream().mapToInt(Integer::intValue).toArray();
            case UINT64 ->
                contents.getUint64ContentsList().stream().mapToLong(Long::longValue).toArray();
            case FP32 ->
                contents.getFp32ContentsList().stream().mapToDouble(Float::doubleValue).toArray();
            case FP64 ->
                contents.getFp64ContentsList().stream().mapToDouble(Double::doubleValue).toArray();
            case BYTES ->
                contents.getBytesContentsList().stream().map(ByteString::toByteArray).toArray(byte[][]::new);
            default ->
                null;
        };
    }

    private boolean[] deserializeBoolArray(ByteBuffer buffer) {
        boolean[] array = new boolean[buffer.remaining()];
        for (int i = 0; i < array.length; i++) {
            array[i] = buffer.get() != 0;
        }
        return array;
    }

    private byte[] deserializeByteArray(ByteBuffer buffer) {
        byte[] array = new byte[buffer.remaining()];
        buffer.get(array);
        return array;
    }

    private short[] deserializeShortArray(ByteBuffer buffer) {
        short[] array = new short[buffer.remaining() / 2];
        buffer.asShortBuffer().get(array);
        return array;
    }

    private int[] deserializeIntArray(ByteBuffer buffer) {
        int[] array = new int[buffer.remaining() / 4];
        buffer.asIntBuffer().get(array);
        return array;
    }

    private long[] deserializeLongArray(ByteBuffer buffer) {
        long[] array = new long[buffer.remaining() / 8];
        buffer.asLongBuffer().get(array);
        return array;
    }

    private short[] deserializeUnsignedByteArray(ByteBuffer buffer) {
        short[] array = new short[buffer.remaining()];
        for (int i = 0; i < array.length; i++) {
            array[i] = (short) (buffer.get() & 0xFF);
        }
        return array;
    }

    private int[] deserializeUnsignedShortArray(ByteBuffer buffer) {
        int[] array = new int[buffer.remaining() / 2];
        for (int i = 0; i < array.length; i++) {
            array[i] = buffer.getShort() & 0xFFFF;
        }
        return array;
    }

    private long[] deserializeUnsignedIntArray(ByteBuffer buffer) {
        long[] array = new long[buffer.remaining() / 4];
        for (int i = 0; i < array.length; i++) {
            array[i] = buffer.getInt() & 0xFFFFFFFFL;
        }
        return array;
    }

    private long[] deserializeUnsignedLongArray(ByteBuffer buffer) {
        long[] array = new long[buffer.remaining() / 8];
        buffer.asLongBuffer().get(array);
        return array;
    }

    private float[] deserializeFloatArray(ByteBuffer buffer) {
        float[] array = new float[buffer.remaining() / 4];
        buffer.asFloatBuffer().get(array);
        return array;
    }

    private double[] deserializeDoubleArray(ByteBuffer buffer) {
        double[] array = new double[buffer.remaining() / 8];
        buffer.asDoubleBuffer().get(array);
        return array;
    }

    private float[] deserializeFP16Array(ByteBuffer buffer) {
        // FP16 (IEEE 754 half-precision) is stored as 2 bytes, convert to float
        float[] array = new float[buffer.remaining() / 2];
        for (int i = 0; i < array.length; i++) {
            short bits = buffer.getShort();
            array[i] = fp16ToFloat(bits);
        }
        return array;
    }

    private float[] deserializeBF16Array(ByteBuffer buffer) {
        // BF16 (Google Brain Float) is stored as 2 bytes, convert to float
        float[] array = new float[buffer.remaining() / 2];
        for (int i = 0; i < array.length; i++) {
            short bits = buffer.getShort();
            array[i] = bfloat16ToFloat(bits);
        }
        return array;
    }

    private String[] deserializeStringArray(ByteBuffer buffer) {
        List<String> strings = new ArrayList<>();

        while (buffer.hasRemaining()) {
            if (buffer.remaining() < 4) {
                break;
            }
            int length = buffer.getInt();
            if (length < 0 || length > buffer.remaining()) {
                throw new TritonInferException(
                        String.format("Truncated string data: expected %d bytes, but only %d available",
                                length, buffer.remaining()));
            }
            byte[] bytes = new byte[length];
            buffer.get(bytes);
            strings.add(new String(bytes, java.nio.charset.StandardCharsets.UTF_8));
        }
        return strings.toArray(String[]::new);
    }

    /**
     * Convert FP16 (IEEE 754 half-precision) to float.
     * <p>
     * FP16 is a 16-bit floating-point format following the IEEE 754 standard
     * with:
     * <ul>
     * <li>1 bit for sign</li>
     * <li>5 bits for exponent (bias = 15)</li>
     * <li>10 bits for mantissa</li>
     * </ul>
     * </p>
     *
     * @param fp16bits The FP16 value as a short
     * @return The float value
     */
    private float fp16ToFloat(short fp16bits) {
        int bits = fp16bits & 0xFFFF;
        int sign = bits & 0x8000;
        int exponent = (bits >> 10) & 0x1F;
        int mantissa = bits & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            } else {
                return Float.intBitsToFloat((sign << 16) | ((exponent + 112) << 23) | (mantissa << 13));
            }
        } else if (exponent == 31) {
            if (mantissa == 0) {
                return Float.intBitsToFloat((sign << 16) | 0x7F800000);
            } else {
                return Float.intBitsToFloat((sign << 16) | 0x7FC00000);
            }
        }
        return Float.intBitsToFloat((sign << 16) | ((exponent + 112) << 23) | (mantissa << 13));
    }

    /**
     * Convert BF16 (Google Brain Float) to float.
     * <p>
     * BF16 is a 16-bit truncated float representation, essentially the 16 most
     * significant bits of a 32-bit IEEE 754 float, with:
     * <ul>
     * <li>1 bit for sign</li>
     * <li>8 bits for exponent (same bias as float32: 127)</li>
     * <li>7 bits for mantissa (truncated from 23 bits)</li>
     * </ul>
     * </p>
     *
     * @param bf16bits The BF16 value as a short
     * @return The float value
     */
    private float bfloat16ToFloat(short bf16bits) {
        return Float.intBitsToFloat(((int) bf16bits & 0xFFFF) << 16);
    }
}
