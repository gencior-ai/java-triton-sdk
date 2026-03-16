package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferRequestedOutput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.TritonDataType;

/**
 * Integration tests for requested output filtering.
 * Validates that the server correctly returns only the requested outputs.
 */
class TritonRequestedOutputIT extends AbstractTritonIntegrationTest {

    // ==================== Sync infer with requested outputs ====================

    @Test
    void infer_withRequestedOutput_shouldReturnOnlyRequestedOutput() {
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.FP32);
        input.setData(new float[]{1.0f, 2.0f, 3.0f});

        List<InferRequestedOutput> outputs = List.of(InferRequestedOutput.of("OUTPUT0"));

        InferResult result = client.infer("identity_fp32", null, List.of(input), outputs, null);

        assertNotNull(result);
        assertEquals("identity_fp32", result.getModelName());
        assertTrue(result.getOutputNames().contains("OUTPUT0"));
        float[] data = result.asFloatArray("OUTPUT0");
        assertEquals(3, data.length);
        assertEquals(1.0f, data[0]);
        assertEquals(2.0f, data[1]);
        assertEquals(3.0f, data[2]);
    }

    @Test
    void infer_withoutRequestedOutput_shouldReturnAllOutputs() {
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.FP32);
        input.setData(new float[]{10.0f, 20.0f});

        InferResult result = client.infer("identity_fp32", List.of(input));

        assertNotNull(result);
        assertTrue(result.getOutputNames().contains("OUTPUT0"));
        float[] data = result.asFloatArray("OUTPUT0");
        assertEquals(2, data.length);
    }

    @Test
    void infer_withNonExistentOutput_shouldFail() {
        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        List<InferRequestedOutput> outputs = List.of(InferRequestedOutput.of("DOES_NOT_EXIST"));

        assertThrows(Exception.class, () ->
                client.infer("identity_fp32", null, List.of(input), outputs, null));
    }

    // ==================== Async infer with requested outputs ====================

    @Test
    void inferAsync_withRequestedOutput_shouldWork() throws Exception {
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.FP32);
        input.setData(new float[]{5.0f, 10.0f});

        List<InferRequestedOutput> outputs = List.of(InferRequestedOutput.of("OUTPUT0"));

        CompletableFuture<InferResult> future = client.inferAsync(
                "identity_fp32", null, List.of(input), outputs, null);

        InferResult result = future.get(30, TimeUnit.SECONDS);

        assertNotNull(result);
        float[] data = result.asFloatArray("OUTPUT0");
        assertEquals(2, data.length);
        assertEquals(5.0f, data[0]);
    }

    // ==================== Adder model with requested outputs ====================

    @Test
    void infer_adderModel_withRequestedOutput_shouldReturnSum() {
        InferInput input0 = new InferInput("INPUT0", new long[]{3}, TritonDataType.INT32);
        input0.setData(new int[]{1, 2, 3});
        InferInput input1 = new InferInput("INPUT1", new long[]{3}, TritonDataType.INT32);
        input1.setData(new int[]{10, 20, 30});

        List<InferRequestedOutput> outputs = List.of(InferRequestedOutput.of("OUTPUT0"));

        InferResult result = client.infer("adder", null, List.of(input0, input1), outputs, null);

        assertNotNull(result);
        int[] data = result.asIntArray("OUTPUT0");
        assertEquals(3, data.length);
        assertEquals(11, data[0]);
        assertEquals(22, data[1]);
        assertEquals(33, data[2]);
    }
}
