package com.gencior.triton.core;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import com.gencior.triton.core.InferResult.OutputTensorDescriptor;
import com.gencior.triton.exceptions.TritonDataNotFoundException;
import com.gencior.triton.exceptions.TritonDataTypeException;
import com.gencior.triton.exceptions.TritonInferException;

/**
 * Tests for the protocol-agnostic constructor of InferResult.
 * Verifies that deserialization works identically to the protobuf path.
 */
public class InferResultAgnosticTest {

    @Test
    public void testGetModelName() {
        InferResult result = createSingleFP32Result("myModel", "2", "req-123", new float[]{1.0f});
        assertEquals("myModel", result.getModelName());
    }

    @Test
    public void testGetModelVersion() {
        InferResult result = createSingleFP32Result("model", "3", "req-456", new float[]{1.0f});
        assertEquals("3", result.getModelVersion());
    }

    @Test
    public void testGetRequestId() {
        InferResult result = createSingleFP32Result("model", "1", "req-789", new float[]{1.0f});
        assertEquals("req-789", result.getRequestId());
    }

    @Test
    public void testGetOutputNames() {
        List<OutputTensorDescriptor> outputs = List.of(
                new OutputTensorDescriptor("output_a", "FP32", new long[]{1}, new byte[4]),
                new OutputTensorDescriptor("output_b", "INT32", new long[]{1}, new byte[4])
        );
        InferResult result = new InferResult("model", "1", "req", outputs);
        List<String> names = result.getOutputNames();
        assertEquals(2, names.size());
        assertEquals("output_a", names.get(0));
        assertEquals("output_b", names.get(1));
    }

    @Test
    public void testGetOutputReturnsNull() {
        InferResult result = createSingleFP32Result("model", "1", "req", new float[]{1.0f});
        assertNull("getOutput() should return null in agnostic mode", result.getOutput("output"));
    }

    @Test
    public void testGetResponseReturnsNull() {
        InferResult result = createSingleFP32Result("model", "1", "req", new float[]{1.0f});
        assertNull("getResponse() should return null in agnostic mode", result.getResponse());
    }

    @Test(expected = NullPointerException.class)
    public void testConstructorNullOutputsThrows() {
        new InferResult("model", "1", "req", null);
    }

    @Test
    public void testFP32Deserialization() {
        float[] expected = {1.5f, 2.5f, 3.5f, -4.5f};
        InferResult result = createSingleFP32Result("model", "1", "req", expected);

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be float[]", output instanceof float[]);
        assertArrayEquals(expected, (float[]) output, 1e-6f);
    }

    @Test
    public void testAsFloatArray() {
        float[] expected = {10.0f, 20.0f};
        InferResult result = createSingleFP32Result("model", "1", "req", expected);
        assertArrayEquals(expected, result.asFloatArray("output"), 1e-6f);
    }

    @Test
    public void testInt32Deserialization() {
        int[] expected = {100, 200, 300, 400};
        byte[] raw = intArrayToBytes(expected);
        InferResult result = createResult("output", "INT32", raw);

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be int[]", output instanceof int[]);
        assertArrayEquals(expected, (int[]) output);
    }

    @Test
    public void testAsIntArray() {
        int[] expected = {42, -7};
        byte[] raw = intArrayToBytes(expected);
        InferResult result = createResult("output", "INT32", raw);
        assertArrayEquals(expected, result.asIntArray("output"));
    }

    @Test
    public void testInt64Deserialization() {
        long[] expected = {100000L, 200000L, -300000L};
        ByteBuffer buf = ByteBuffer.allocate(expected.length * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (long v : expected) buf.putLong(v);
        InferResult result = createResult("output", "INT64", buf.array());

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be long[]", output instanceof long[]);
        assertArrayEquals(expected, (long[]) output);
    }

    @Test
    public void testAsLongArray() {
        long[] expected = {1L, 2L};
        ByteBuffer buf = ByteBuffer.allocate(expected.length * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (long v : expected) buf.putLong(v);
        InferResult result = createResult("output", "INT64", buf.array());
        assertArrayEquals(expected, result.asLongArray("output"));
    }

    @Test
    public void testFP64Deserialization() {
        double[] expected = {1.1, 2.2, 3.3};
        ByteBuffer buf = ByteBuffer.allocate(expected.length * Double.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : expected) buf.putDouble(v);
        InferResult result = createResult("output", "FP64", buf.array());

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be double[]", output instanceof double[]);
        assertArrayEquals(expected, (double[]) output, 1e-10);
    }

    @Test
    public void testAsDoubleArray() {
        double[] expected = {3.14};
        ByteBuffer buf = ByteBuffer.allocate(Double.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buf.putDouble(expected[0]);
        InferResult result = createResult("output", "FP64", buf.array());
        assertArrayEquals(expected, result.asDoubleArray("output"), 1e-10);
    }

    @Test
    public void testBytesDeserialization() {
        String[] expected = {"hello", "world"};
        byte[] raw = serializeStrings(expected);
        InferResult result = createResult("output", "BYTES", raw);

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be String[]", output instanceof String[]);
        assertArrayEquals(expected, (String[]) output);
    }

    @Test
    public void testAsStringArray() {
        String[] expected = {"foo", "bar", "baz"};
        byte[] raw = serializeStrings(expected);
        InferResult result = createResult("output", "BYTES", raw);
        assertArrayEquals(expected, result.asStringArray("output"));
    }

    @Test
    public void testBoolDeserialization() {
        byte[] raw = {1, 0, 1, 1, 0};
        InferResult result = createResult("output", "BOOL", raw);

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be boolean[]", output instanceof boolean[]);
        boolean[] expected = {true, false, true, true, false};
        boolean[] actual = (boolean[]) output;
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("Mismatch at index " + i, expected[i], actual[i]);
        }
    }

    @Test
    public void testInt16Deserialization() {
        short[] expected = {10, 20, -30};
        ByteBuffer buf = ByteBuffer.allocate(expected.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (short v : expected) buf.putShort(v);
        InferResult result = createResult("output", "INT16", buf.array());

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be short[]", output instanceof short[]);
        assertArrayEquals(expected, (short[]) output);
    }

    @Test
    public void testInt8Deserialization() {
        byte[] expected = {1, 2, -3, 4};
        InferResult result = createResult("output", "INT8", expected);

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be byte[]", output instanceof byte[]);
        assertArrayEquals(expected, (byte[]) output);
    }

    @Test
    public void testUint8Deserialization() {
        byte[] raw = {(byte) 0, (byte) 127, (byte) 255};
        InferResult result = createResult("output", "UINT8", raw);

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be short[]", output instanceof short[]);
        short[] expected = {0, 127, 255};
        assertArrayEquals(expected, (short[]) output);
    }

    @Test
    public void testUint16Deserialization() {
        ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort((short) 100);
        buf.putShort((short) -1); // 65535 unsigned
        InferResult result = createResult("output", "UINT16", buf.array());

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be int[]", output instanceof int[]);
        int[] expected = {100, 65535};
        assertArrayEquals(expected, (int[]) output);
    }

    @Test
    public void testUint32Deserialization() {
        ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(100);
        buf.putInt(-1); // 4294967295 unsigned
        InferResult result = createResult("output", "UINT32", buf.array());

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be long[]", output instanceof long[]);
        long[] expected = {100L, 4294967295L};
        assertArrayEquals(expected, (long[]) output);
    }

    @Test
    public void testMultipleOutputs() {
        float[] floats = {1.0f, 2.0f};
        int[] ints = {10, 20, 30};
        List<OutputTensorDescriptor> outputs = List.of(
                new OutputTensorDescriptor("float_out", "FP32", new long[]{2}, floatArrayToBytes(floats)),
                new OutputTensorDescriptor("int_out", "INT32", new long[]{3}, intArrayToBytes(ints))
        );
        InferResult result = new InferResult("model", "1", "req", outputs);

        assertArrayEquals(floats, result.asFloatArray("float_out"), 1e-6f);
        assertArrayEquals(ints, result.asIntArray("int_out"));
    }

    @Test(expected = TritonDataNotFoundException.class)
    public void testOutputNotFound() {
        InferResult result = createSingleFP32Result("model", "1", "req", new float[]{1.0f});
        result.getOutputAsArray("nonexistent");
    }

    @Test(expected = TritonInferException.class)
    public void testEmptyRawContent() {
        InferResult result = createResult("output", "FP32", new byte[0]);
        result.getOutputAsArray("output");
    }

    @Test(expected = TritonInferException.class)
    public void testNullRawContent() {
        List<OutputTensorDescriptor> outputs = List.of(
                new OutputTensorDescriptor("output", "FP32", new long[]{1}, null)
        );
        InferResult result = new InferResult("model", "1", "req", outputs);
        result.getOutputAsArray("output");
    }

    @Test(expected = TritonDataTypeException.class)
    public void testInvalidDatatype() {
        InferResult result = createResult("output", "INVALID_TYPE", new byte[]{1, 2, 3, 4});
        result.getOutputAsArray("output");
    }

    @Test
    public void testSingleElementFP32() {
        float[] expected = {42.0f};
        InferResult result = createSingleFP32Result("model", "1", "req", expected);
        assertArrayEquals(expected, result.asFloatArray("output"), 1e-6f);
    }

    @Test
    public void testSingleElementInt32() {
        int[] expected = {-1};
        byte[] raw = intArrayToBytes(expected);
        InferResult result = createResult("output", "INT32", raw);
        assertArrayEquals(expected, result.asIntArray("output"));
    }

    @Test
    public void testFP32SpecialValues() {
        float[] expected = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0.0f, -0.0f};
        InferResult result = createSingleFP32Result("model", "1", "req", expected);
        float[] actual = result.asFloatArray("output");
        assertEquals(expected.length, actual.length);
        assertTrue(Float.isNaN(actual[0]));
        assertEquals(Float.POSITIVE_INFINITY, actual[1], 0);
        assertEquals(Float.NEGATIVE_INFINITY, actual[2], 0);
        assertEquals(0.0f, actual[3], 0);
    }

    @Test
    public void testFP64SpecialValues() {
        double[] expected = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        ByteBuffer buf = ByteBuffer.allocate(expected.length * Double.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : expected) buf.putDouble(v);
        InferResult result = createResult("output", "FP64", buf.array());
        double[] actual = result.asDoubleArray("output");
        assertTrue(Double.isNaN(actual[0]));
        assertEquals(Double.POSITIVE_INFINITY, actual[1], 0);
        assertEquals(Double.NEGATIVE_INFINITY, actual[2], 0);
    }

    @Test
    public void testBytesUnicodeStrings() {
        String[] expected = {"hello", "monde", "\u00e9\u00e0\u00fc", "\u4e16\u754c", ""};
        byte[] raw = serializeStrings(expected);
        InferResult result = createResult("output", "BYTES", raw);
        assertArrayEquals(expected, result.asStringArray("output"));
    }

    @Test
    public void testBytesSingleEmptyString() {
        String[] expected = {""};
        byte[] raw = serializeStrings(expected);
        InferResult result = createResult("output", "BYTES", raw);
        assertArrayEquals(expected, result.asStringArray("output"));
    }

    @Test
    public void testUint64Deserialization() {
        ByteBuffer buf = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(0L);
        buf.putLong(Long.MAX_VALUE);
        InferResult result = createResult("output", "UINT64", buf.array());

        Object output = result.getOutputAsArray("output");
        assertTrue("Should be long[]", output instanceof long[]);
        long[] actual = (long[]) output;
        assertEquals(0L, actual[0]);
        assertEquals(Long.MAX_VALUE, actual[1]);
    }

    @Test
    public void testNullModelNameAndVersion() {
        List<OutputTensorDescriptor> outputs = List.of(
                new OutputTensorDescriptor("out", "FP32", new long[]{1}, floatArrayToBytes(new float[]{1.0f}))
        );
        InferResult result = new InferResult(null, null, null, outputs);
        assertNull(result.getModelName());
        assertNull(result.getModelVersion());
        assertNull(result.getRequestId());
    }

    @Test(expected = TritonDataTypeException.class)
    public void testInt32InvalidBufferSize() {
        // 3 bytes is not a valid INT32 buffer (must be multiple of 4)
        InferResult result = createResult("output", "INT32", new byte[]{1, 2, 3});
        result.getOutputAsArray("output");
    }

    @Test(expected = TritonDataTypeException.class)
    public void testFP64InvalidBufferSize() {
        // 5 bytes is not a valid FP64 buffer (must be multiple of 8)
        InferResult result = createResult("output", "FP64", new byte[]{1, 2, 3, 4, 5});
        result.getOutputAsArray("output");
    }

    @Test
    public void testOutputNamesOrder() {
        List<OutputTensorDescriptor> outputs = List.of(
                new OutputTensorDescriptor("first", "FP32", new long[]{1}, floatArrayToBytes(new float[]{1.0f})),
                new OutputTensorDescriptor("second", "INT32", new long[]{1}, intArrayToBytes(new int[]{1})),
                new OutputTensorDescriptor("third", "BYTES", new long[]{1}, serializeStrings(new String[]{"x"}))
        );
        InferResult result = new InferResult("model", "1", "req", outputs);
        List<String> names = result.getOutputNames();
        assertEquals(3, names.size());
        assertEquals("first", names.get(0));
        assertEquals("second", names.get(1));
        assertEquals("third", names.get(2));
    }

    private InferResult createSingleFP32Result(String modelName, String version, String reqId, float[] data) {
        byte[] raw = floatArrayToBytes(data);
        List<OutputTensorDescriptor> outputs = List.of(
                new OutputTensorDescriptor("output", "FP32", new long[]{data.length}, raw)
        );
        return new InferResult(modelName, version, reqId, outputs);
    }

    private InferResult createResult(String outputName, String datatype, byte[] rawContent) {
        List<OutputTensorDescriptor> outputs = List.of(
                new OutputTensorDescriptor(outputName, datatype, new long[]{}, rawContent)
        );
        return new InferResult("model", "1", "req", outputs);
    }

    private byte[] floatArrayToBytes(float[] data) {
        ByteBuffer buf = ByteBuffer.allocate(data.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) buf.putFloat(v);
        return buf.array();
    }

    private byte[] intArrayToBytes(int[] data) {
        ByteBuffer buf = ByteBuffer.allocate(data.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : data) buf.putInt(v);
        return buf.array();
    }

    private byte[] serializeStrings(String[] strings) {
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
}
