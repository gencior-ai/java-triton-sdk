package com.gencior.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.nio.ByteBuffer;

import org.junit.Before;
import org.junit.Test;

import com.gencior.core.InferResult;
import com.gencior.exceptions.TritonDataNotFoundException;
import com.gencior.exceptions.TritonDataTypeException;
import com.gencior.exceptions.TritonInferException;
import com.google.protobuf.ByteString;

import inference.GrpcService.ModelInferResponse;
import inference.GrpcService.ModelInferResponse.InferOutputTensor;

public class InferResultTest {

    private InferResult inferResult;

    @Before
    public void setUp() {
    }

    @Test
    public void testInt32FromRawContent() {
        int[] expectedData = {100, 200, 300, 400};
        ByteBuffer buffer = ByteBuffer.allocate(4 * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int value : expectedData) {
            buffer.putInt(value);
        }
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "INT32", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be int[]", result instanceof int[]);
        assertArrayEquals("INT32 raw content should deserialize to int[]", expectedData, (int[]) result);
    }

    /**
     * Test INT16 deserialization from raw content
     * deserializeRawContent returns short[] (CORRECT)
     */
    @Test
    public void testInt16FromRawContent() {
        short[] expectedData = {100, 200, 300};
        ByteBuffer buffer = ByteBuffer.allocate(2 * 3).order(java.nio.ByteOrder.LITTLE_ENDIAN);;
        for (short value : expectedData) {
            buffer.putShort(value);
        }
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "INT16", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be short[]", result instanceof short[]);
        assertArrayEquals("INT16 raw content should deserialize to short[]", expectedData, (short[]) result);
    }

    /**
     * Test INT8 deserialization from raw content
     * deserializeRawContent returns byte[] (CORRECT)
     */
    @Test
    public void testInt8FromRawContent() {
        byte[] expectedData = {1, 2, 3, 4, 5};
        ByteBuffer buffer = ByteBuffer.wrap(expectedData).order(java.nio.ByteOrder.LITTLE_ENDIAN);

        ModelInferResponse response = createResponseWithRawContent("output", "INT8", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be byte[]", result instanceof byte[]);
        assertArrayEquals("INT8 raw content should deserialize to byte[]", expectedData, (byte[]) result);
    }

    /**
     * Test INT64 deserialization from raw content
     * deserializeRawContent returns long[] (CORRECT)
     */
    @Test
    public void testInt64FromRawContent() {
        long[] expectedData = {1000000L, 2000000L, 3000000L};
        ByteBuffer buffer = ByteBuffer.allocate(8 * 3).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (long value : expectedData) {
            buffer.putLong(value);
        }
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "INT64", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be long[]", result instanceof long[]);
        assertArrayEquals("INT64 raw content should deserialize to long[]", expectedData, (long[]) result);
    }

    // ========== TESTS FOR UNSIGNED INTEGER TYPES - TYPE MISMATCH ISSUES ==========

    /**
     * Test UINT8 deserialization from raw content
     * deserializeRawContent returns short[] (CORRECT - needed because byte is signed in Java)
     */
    @Test
    public void testUint8FromRawContent() {
        byte[] byteData = {(byte) 255, (byte) 128, (byte) 64};
        ByteBuffer buffer = ByteBuffer.wrap(byteData).order(java.nio.ByteOrder.LITTLE_ENDIAN);

        ModelInferResponse response = createResponseWithRawContent("output", "UINT8", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be short[]", result instanceof short[]);
        short[] shortArray = (short[]) result;
        assertEquals("UINT8 255 should map to short 255", 255, shortArray[0]);
        assertEquals("UINT8 128 should map to short 128", 128, shortArray[1]);
    }

    /**
     * Test UINT16 deserialization from raw content
     * deserializeRawContent returns int[] (CORRECT - needed for unsigned short values up to 65535)
     */
    @Test
    public void testUint16FromRawContent() {
        ByteBuffer buffer = ByteBuffer.allocate(2 * 3).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putShort((short) 65535); // Max unsigned short
        buffer.putShort((short) 32768); // Mid range
        buffer.putShort((short) 0);
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "UINT16", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be int[]", result instanceof int[]);
        int[] intArray = (int[]) result;
        assertEquals("UINT16 65535 should map to int 65535", 65535, intArray[0]);
        assertEquals("UINT16 32768 should map to int 32768", 32768, intArray[1]);
    }

    /**
     * Test UINT32 deserialization from raw content
     * deserializeRawContent returns long[] (CORRECT)
     */
    @Test
    public void testUint32FromRawContent() {
        ByteBuffer buffer = ByteBuffer.allocate(4 * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putInt((int) 0xFFFFFFFF); // Max unsigned int as signed int
        buffer.putInt(0);
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "UINT32", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be long[]", result instanceof long[]);
        long[] longArray = (long[]) result;
        assertEquals("UINT32 0xFFFFFFFF should map to long 4294967295", 4294967295L, longArray[0]);
    }

    /**
     * Test UINT64 deserialization from raw content
     * deserializeRawContent returns long[] (but values may overflow for actual UINT64 max)
     */
    @Test
    public void testUint64FromRawContent() {
        ByteBuffer buffer = ByteBuffer.allocate(8 * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(Long.MAX_VALUE);
        buffer.putLong(0);
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "UINT64", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be long[]", result instanceof long[]);
        long[] longArray = (long[]) result;
        assertEquals("First value should be Long.MAX_VALUE", Long.MAX_VALUE, longArray[0]);
    }

    // ========== TESTS FOR FLOATING POINT TYPES ==========

    /**
     * Test FP32 deserialization from raw content
     * deserializeRawContent returns float[] (CORRECT)
     */
    @Test
    public void testFp32FromRawContent() {
        float[] expectedData = {1.5f, 2.5f, 3.5f};
        ByteBuffer buffer = ByteBuffer.allocate(4 * 3).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float value : expectedData) {
            buffer.putFloat(value);
        }
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "FP32", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be float[]", result instanceof float[]);
        assertArrayEquals("FP32 raw content should deserialize to float[]", expectedData, (float[]) result, 0.001f);
    }

    /**
     * Test FP64 deserialization from raw content
     * deserializeRawContent returns double[] (CORRECT)
     */
    @Test
    public void testFp64FromRawContent() {
        double[] expectedData = {1.123, 2.456, 3.789};
        ByteBuffer buffer = ByteBuffer.allocate(8 * 3).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (double value : expectedData) {
            buffer.putDouble(value);
        }
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "FP64", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be double[]", result instanceof double[]);
        assertArrayEquals("FP64 raw content should deserialize to double[]", expectedData, (double[]) result, 0.000001);
    }

    /**
     * Test FP16 deserialization from raw content
     * deserializeRawContent returns float[] (CORRECT)
     */
    @Test
    public void testFp16FromRawContent() {
        // FP16 representation of simple values
        ByteBuffer buffer = ByteBuffer.allocate(2 * 3).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putShort((short) 0x3C00); // FP16 for 1.0
        buffer.putShort((short) 0x4000); // FP16 for 2.0
        buffer.putShort((short) 0x4400); // FP16 for 3.0
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "FP16", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be float[]", result instanceof float[]);
        float[] floatArray = (float[]) result;
        assertEquals("FP16 0x3C00 should convert to ~1.0", 1.0f, floatArray[0], 0.001f);
    }

    /**
     * Test BF16 deserialization from raw content
     * deserializeRawContent returns float[] (CORRECT)
     */
    @Test
    public void testBf16FromRawContent() {
        // BF16 is just truncated FP32
        ByteBuffer buffer = ByteBuffer.allocate(2 * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putShort((short) 0x3F80); // BF16 for 1.0
        buffer.putShort((short) 0x4000); // BF16 for 2.0
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "BF16", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be float[]", result instanceof float[]);
        float[] floatArray = (float[]) result;
        assertEquals("BF16 should deserialize to float array", 2, floatArray.length);
    }

    // ========== TESTS FOR BOOL AND BYTES ==========

    /**
     * Test BOOL deserialization from raw content
     * deserializeRawContent returns boolean[] (CORRECT)
     */
    @Test
    public void testBoolFromRawContent() {
        byte[] boolData = {1, 0, 1, 1, 0};
        ByteBuffer buffer = ByteBuffer.wrap(boolData).order(java.nio.ByteOrder.LITTLE_ENDIAN);

        ModelInferResponse response = createResponseWithRawContent("output", "BOOL", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be boolean[]", result instanceof boolean[]);
        boolean[] boolArray = (boolean[]) result;
        assertEquals("First bool should be true", true, boolArray[0]);
        assertEquals("Second bool should be false", false, boolArray[1]);
    }

    /**
     * Test BYTES deserialization from raw content
     * deserializeRawContent returns String[] (CORRECT)
     */
    @Test
    public void testBytesFromRawContent() {
        // BYTES format: length (4 bytes) + data + length + data...
        ByteBuffer buffer = ByteBuffer.allocate(100).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(5);
        buffer.put("hello".getBytes());
        buffer.putInt(5);
        buffer.put("world".getBytes());
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "BYTES", buffer);
        inferResult = new InferResult(response);

        Object result = inferResult.getOutputAsArray("output");
        assertTrue("Result should be String[]", result instanceof String[]);
        String[] stringArray = (String[]) result;
        assertEquals("Should have 2 strings", 2, stringArray.length);
        assertEquals("First string should be 'hello'", "hello", stringArray[0]);
        assertEquals("Second string should be 'world'", "world", stringArray[1]);
    }

    // ========== TESTS FOR OUTPUT NOT FOUND ==========

    @Test(expected = TritonDataNotFoundException.class)
    public void testOutputNotFound() {
        ModelInferResponse response = ModelInferResponse.newBuilder()
            .build();

        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("nonexistent");
    }

    @Test(expected = TritonInferException.class)
    public void testOutputFoundButNoData() {
        ModelInferResponse response = ModelInferResponse.newBuilder()
            .addOutputs(InferOutputTensor.newBuilder()
                .setName("empty_output")
                .setDatatype("FP32")
                .build())
            .build();

        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("empty_output");
    }

    // ========== TESTS FOR INVALID DATATYPES ==========

    @Test(expected = TritonDataTypeException.class)
    public void testInvalidDatatype() {
        ByteBuffer buffer = ByteBuffer.allocate(10).order(java.nio.ByteOrder.LITTLE_ENDIAN);

        ModelInferResponse response = createResponseWithRawContent("output", "INVALID_TYPE", buffer);
        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("output");
    }

    // ========== TESTS FOR BUFFER SIZE VALIDATION ==========

    @Test(expected = TritonDataTypeException.class)
    public void testInvalidBufferSizeForInt32() {
        // INT32 needs multiple of 4 bytes
        ByteBuffer buffer = ByteBuffer.allocate(3).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.put((byte) 1);
        buffer.put((byte) 2);
        buffer.put((byte) 3);
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "INT32", buffer);
        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("output");
    }

    @Test(expected = TritonDataTypeException.class)
    public void testInvalidBufferSizeForInt16() {
        // INT16 needs multiple of 2 bytes
        ByteBuffer buffer = ByteBuffer.allocate(3).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.put((byte) 1);
        buffer.put((byte) 2);
        buffer.put((byte) 3);
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "INT16", buffer);
        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("output");
    }

    @Test(expected = TritonDataTypeException.class)
    public void testInvalidBufferSizeForInt64() {
        // INT64 needs multiple of 8 bytes
        ByteBuffer buffer = ByteBuffer.allocate(7).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 7; i++) {
            buffer.put((byte) i);
        }
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "INT64", buffer);
        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("output");
    }

    // ========== TESTS FOR INVALID STRING DATA ==========

    @Test(expected = TritonInferException.class)
    public void testInvalidStringLengthNegative() {
        // String with negative length is invalid
        ByteBuffer buffer = ByteBuffer.allocate(4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(-1); // Invalid: negative length
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "BYTES", buffer);
        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("output");
    }

    @Test(expected = TritonInferException.class)
    public void testTruncatedStringData() {
        // String length indicates more data than available
        ByteBuffer buffer = ByteBuffer.allocate(10).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(100); // Says 100 bytes coming
        buffer.put((byte) 1); // But only 1 byte available
        buffer.flip();

        ModelInferResponse response = createResponseWithRawContent("output", "BYTES", buffer);
        inferResult = new InferResult(response);
        inferResult.getOutputAsArray("output");
    }

    // ========== TESTS FOR getOutput METHOD ==========

    @Test
    public void testGetOutputByName() {
        ModelInferResponse response = ModelInferResponse.newBuilder()
            .addOutputs(InferOutputTensor.newBuilder()
                .setName("output1")
                .setDatatype("FP32")
                .build())
            .addOutputs(InferOutputTensor.newBuilder()
                .setName("output2")
                .setDatatype("INT32")
                .build())
            .build();

        inferResult = new InferResult(response);
        InferOutputTensor output = inferResult.getOutput("output2");

        assertNotNull("Should find output by name", output);
        assertEquals("Output name should match", "output2", output.getName());
        assertEquals("Output datatype should match", "INT32", output.getDatatype());
    }

    @Test
    public void testGetOutputNotFound() {
        ModelInferResponse response = ModelInferResponse.newBuilder()
            .build();

        inferResult = new InferResult(response);
        InferOutputTensor output = inferResult.getOutput("nonexistent");

        assertEquals("Should return null for nonexistent output", null, output);
    }

    // ========== TESTS FOR getResponse METHOD ==========

    @Test
    public void testGetResponse() {
        ModelInferResponse response = ModelInferResponse.newBuilder()
            .setModelName("test_model")
            .build();

        inferResult = new InferResult(response);
        ModelInferResponse retrievedResponse = inferResult.getResponse();

        assertNotNull("Should return response", retrievedResponse);
        assertEquals("Model name should match", "test_model", retrievedResponse.getModelName());
    }

    // ========== HELPER METHODS ==========

    /**
     * Helper method to create a ModelInferResponse with raw content
     */
    private ModelInferResponse createResponseWithRawContent(String outputName, String datatype, ByteBuffer buffer) {
        return ModelInferResponse.newBuilder()
            .addOutputs(InferOutputTensor.newBuilder()
                .setName(outputName)
                .setDatatype(datatype)
                .build())
            .addRawOutputContents(ByteString.copyFrom(buffer))
            .build();
    }
}
