package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferParameters;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.TritonDataType;
import com.gencior.triton.exceptions.TritonDataNotFoundException;

class TritonInferenceIT extends AbstractTritonIntegrationTest {

    // ==================== Synchronous Inference ====================

    @Test
    void infer_fp32Identity_shouldEchoInput() {
        float[] inputData = {1.0f, 2.5f, 3.7f, -4.2f, 0.0f};
        InferInput input = new InferInput("INPUT0", new long[]{5}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = client.infer("identity_fp32", List.of(input));

        assertNotNull(result);
        assertEquals("identity_fp32", result.getModelName());
        assertNotNull(result.getModelVersion());
        assertTrue(result.getOutputNames().contains("OUTPUT0"));

        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void infer_int32Identity_shouldEchoInput() {
        int[] inputData = {1, -42, 0, Integer.MAX_VALUE, Integer.MIN_VALUE};
        InferInput input = new InferInput("INPUT0", new long[]{5}, TritonDataType.INT32);
        input.setData(inputData);

        InferResult result = client.infer("identity_int32", List.of(input));

        assertNotNull(result);
        int[] output = result.asIntArray("OUTPUT0");
        assertArrayEquals(inputData, output);
    }

    @Test
    void infer_stringIdentity_shouldEchoInput() {
        String[] inputData = {"hello", "world", "triton", "sdk"};
        InferInput input = new InferInput("INPUT0", new long[]{4}, TritonDataType.BYTES);
        input.setData(inputData);

        InferResult result = client.infer("identity_string", List.of(input));

        assertNotNull(result);
        String[] output = result.asStringArray("OUTPUT0");
        assertArrayEquals(inputData, output);
    }

    @Test
    void infer_adder_shouldSumTwoInputs() {
        int[] data0 = {10, 20, 30};
        int[] data1 = {1, 2, 3};
        InferInput input0 = new InferInput("INPUT0", new long[]{3}, TritonDataType.INT32);
        input0.setData(data0);
        InferInput input1 = new InferInput("INPUT1", new long[]{3}, TritonDataType.INT32);
        input1.setData(data1);

        InferResult result = client.infer("adder", List.of(input0, input1));

        assertNotNull(result);
        int[] output = result.asIntArray("OUTPUT0");
        assertArrayEquals(new int[]{11, 22, 33}, output);
    }

    @Test
    void infer_withExplicitVersion_shouldWork() {
        float[] inputData = {1.0f, 2.0f};
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = client.infer("identity_fp32", "1", List.of(input), null);

        assertNotNull(result);
        assertEquals("1", result.getModelVersion());
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void infer_withCustomParameters_shouldWork() {
        float[] inputData = {5.0f};
        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(inputData);

        InferParameters params = new InferParameters.Builder()
                .add("custom_key", "custom_value")
                .build();

        InferResult result = client.infer("identity_fp32", "1", List.of(input), params);

        assertNotNull(result);
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void infer_withSingleElement_shouldWork() {
        float[] inputData = {42.0f};
        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = client.infer("identity_fp32", List.of(input));

        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void infer_withLargeArray_shouldWork() {
        float[] inputData = new float[1000];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = i * 0.1f;
        }
        InferInput input = new InferInput("INPUT0", new long[]{1000}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = client.infer("identity_fp32", List.of(input));

        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-5f);
    }

    @Test
    void infer_withoutData_shouldThrowException() {
        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);

        assertThrows(TritonDataNotFoundException.class, () ->
                client.infer("identity_fp32", List.of(input))
        );
    }

    // ==================== InferResult Accessors ====================

    @Test
    void inferResult_getOutput_shouldReturnTensor() {
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.FP32);
        input.setData(new float[]{1.0f, 2.0f});

        InferResult result = client.infer("identity_fp32", List.of(input));

        assertNotNull(result.getOutput("OUTPUT0"));
        assertEquals("OUTPUT0", result.getOutput("OUTPUT0").getName());
        assertEquals("FP32", result.getOutput("OUTPUT0").getDatatype());
    }

    @Test
    void inferResult_getResponse_shouldReturnFullProtobuf() {
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.INT32);
        input.setData(new int[]{10, 20});

        InferResult result = client.infer("identity_int32", List.of(input));

        assertNotNull(result.getResponse());
        assertEquals("identity_int32", result.getResponse().getModelName());
    }

    @Test
    void inferResult_getOutputNames_shouldListAllOutputs() {
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.FP32);
        input.setData(new float[]{1.0f, 2.0f});

        InferResult result = client.infer("identity_fp32", List.of(input));

        List<String> names = result.getOutputNames();
        assertEquals(1, names.size());
        assertEquals("OUTPUT0", names.get(0));
    }

    // ==================== Asynchronous Inference ====================

    @Test
    void inferAsync_fp32_shouldReturnFuture() throws Exception {
        float[] inputData = {1.0f, 2.0f, 3.0f};
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.FP32);
        input.setData(inputData);

        CompletableFuture<InferResult> future = client.inferAsync("identity_fp32", List.of(input));

        assertNotNull(future);
        InferResult result = future.get(10, TimeUnit.SECONDS);
        assertNotNull(result);
        assertEquals("identity_fp32", result.getModelName());
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void inferAsync_int32_shouldReturnCorrectResult() throws Exception {
        int[] inputData = {100, 200, 300};
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.INT32);
        input.setData(inputData);

        CompletableFuture<InferResult> future = client.inferAsync("identity_int32", List.of(input));
        InferResult result = future.get(10, TimeUnit.SECONDS);

        int[] output = result.asIntArray("OUTPUT0");
        assertArrayEquals(inputData, output);
    }

    @Test
    void inferAsync_withVersionAndParams_shouldWork() throws Exception {
        float[] inputData = {7.0f, 8.0f};
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.FP32);
        input.setData(inputData);

        InferParameters params = new InferParameters.Builder()
                .add("test_param", 42L)
                .build();

        CompletableFuture<InferResult> future = client.inferAsync("identity_fp32", "1", List.of(input), params);
        InferResult result = future.get(10, TimeUnit.SECONDS);

        assertNotNull(result);
        assertEquals("1", result.getModelVersion());
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void inferAsync_multipleConcurrent_shouldAllSucceed() throws Exception {
        int concurrency = 5;
        @SuppressWarnings("unchecked")
        CompletableFuture<InferResult>[] futures = new CompletableFuture[concurrency];

        for (int i = 0; i < concurrency; i++) {
            float[] inputData = {(float) i};
            InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
            input.setData(inputData);
            futures[i] = client.inferAsync("identity_fp32", List.of(input));
        }

        CompletableFuture.allOf(futures).get(30, TimeUnit.SECONDS);

        for (int i = 0; i < concurrency; i++) {
            InferResult result = futures[i].get();
            assertNotNull(result);
            float[] output = result.asFloatArray("OUTPUT0");
            assertEquals((float) i, output[0], 1e-6f);
        }
    }

    @Test
    void inferAsync_withoutData_shouldCompleteExceptionally() {
        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);

        CompletableFuture<InferResult> future = client.inferAsync("identity_fp32", List.of(input));

        assertTrue(future.isCompletedExceptionally());
    }

    // ==================== Sleeper Model (Delay) ====================

    @Test
    void infer_sleeperModel_shouldReturnAfterDelay() {
        float[] inputData = {99.0f};
        InferInput input0 = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input0.setData(inputData);
        InferInput delayInput = new InferInput("DELAY_MS", new long[]{1}, TritonDataType.INT32);
        delayInput.setData(new int[]{500});

        long start = System.currentTimeMillis();
        InferResult result = client.infer("sleeper", List.of(input0, delayInput));
        long elapsed = System.currentTimeMillis() - start;

        assertNotNull(result);
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
        assertTrue(elapsed >= 400, "Expected at least 400ms delay, got " + elapsed + "ms");
    }

    @Test
    void inferAsync_sleeperModel_shouldReturnAfterDelay() throws Exception {
        float[] inputData = {77.0f};
        InferInput input0 = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input0.setData(inputData);
        InferInput delayInput = new InferInput("DELAY_MS", new long[]{1}, TritonDataType.INT32);
        delayInput.setData(new int[]{300});

        CompletableFuture<InferResult> future = client.inferAsync("sleeper", List.of(input0, delayInput));
        InferResult result = future.get(10, TimeUnit.SECONDS);

        assertNotNull(result);
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }
}
