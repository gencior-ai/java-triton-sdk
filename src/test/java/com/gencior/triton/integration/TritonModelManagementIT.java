package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;

import com.gencior.triton.core.pojo.TritonModelConfig;
import com.gencior.triton.core.pojo.TritonModelMetadata;
import com.gencior.triton.core.pojo.TritonModelStatistics;
import com.gencior.triton.core.pojo.TritonRepositoryIndex;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class TritonModelManagementIT extends AbstractTritonIntegrationTest {

    private static final String MODEL_ID = "identity_fp32";

    // --- Model Ready ---

    @Test
    @Order(1)
    void isModelReady_withModelId_shouldReturnTrue() {
        assertTrue(client.isModelReady(MODEL_ID));
    }

    @Test
    @Order(2)
    void isModelReady_withModelIdAndVersion_shouldReturnTrue() {
        assertTrue(client.isModelReady(MODEL_ID, "1"));
    }

    // --- Model Metadata ---

    @Test
    @Order(3)
    void getModelMetadata_shouldReturnInputAndOutputSchema() {
        TritonModelMetadata metadata = client.getModelMetadata(MODEL_ID, "1");

        assertNotNull(metadata);
        assertEquals(MODEL_ID, metadata.getName());
        assertFalse(metadata.getInputs().isEmpty());
        assertFalse(metadata.getOutputs().isEmpty());
        assertEquals("INPUT0", metadata.getInputs().get(0).getName());
        assertEquals("FP32", metadata.getInputs().get(0).getDatatype());
        assertEquals("OUTPUT0", metadata.getOutputs().get(0).getName());
        assertEquals("FP32", metadata.getOutputs().get(0).getDatatype());
    }

    @Test
    @Order(4)
    void getModelMetadata_withNullVersion_shouldReturnLatest() {
        TritonModelMetadata metadata = client.getModelMetadata(MODEL_ID, null);

        assertNotNull(metadata);
        assertEquals(MODEL_ID, metadata.getName());
    }

    // --- Model Config ---

    @Test
    @Order(5)
    void getModelConfig_shouldReturnValidConfig() {
        TritonModelConfig config = client.getModelConfig(MODEL_ID);

        assertNotNull(config);
        assertEquals(MODEL_ID, config.getName());
        assertEquals("python", config.getBackend());
    }

    @Test
    @Order(6)
    void getModelConfig_withVersion_shouldReturnValidConfig() {
        TritonModelConfig config = client.getModelConfig(MODEL_ID, "1");

        assertNotNull(config);
        assertEquals(MODEL_ID, config.getName());
    }

    // --- Repository Index ---

    @Test
    @Order(7)
    void getModelRepositoryIndex_shouldContainAllModels() {
        TritonRepositoryIndex index = client.getModelRepositoryIndex();

        assertNotNull(index);
        assertNotNull(index.getModels());
        assertTrue(index.getModels().size() >= 5);

        List<String> modelNames = index.getModels().stream()
                .map(m -> m.getName())
                .toList();
        assertTrue(modelNames.contains("identity_fp32"));
        assertTrue(modelNames.contains("identity_int32"));
        assertTrue(modelNames.contains("identity_string"));
        assertTrue(modelNames.contains("adder"));
        assertTrue(modelNames.contains("sleeper"));
    }

    // --- Unload / Load ---

    @Test
    @Order(10)
    void unLoadModel_shouldMakeModelUnavailable() {
        client.unLoadModel("sleeper");

        assertFalse(client.isModelReady("sleeper"));
    }

    @Test
    @Order(11)
    void loadModel_shouldMakeModelAvailableAgain() {
        client.loadModel("sleeper");

        assertTrue(client.isModelReady("sleeper"));
    }

    // --- Inference Statistics ---

    @Test
    @Order(20)
    void getInferenceStatistics_shouldReturnStats() {
        // Trigger at least one inference first
        var input = new com.gencior.triton.core.InferInput("INPUT0", new long[]{3}, com.gencior.triton.core.TritonDataType.FP32);
        input.setData(new float[]{1.0f, 2.0f, 3.0f});
        client.infer(MODEL_ID, List.of(input));

        List<TritonModelStatistics> stats = client.getInferenceStatistics(MODEL_ID, "1");

        assertNotNull(stats);
        assertFalse(stats.isEmpty());
        assertEquals(MODEL_ID, stats.get(0).getName());
        assertTrue(stats.get(0).getInferenceCount() >= 1);
    }

    @Test
    @Order(21)
    void getInferenceStatistics_withNullParams_shouldReturnAllStats() {
        List<TritonModelStatistics> stats = client.getInferenceStatistics(null, null);

        assertNotNull(stats);
        assertFalse(stats.isEmpty());
    }
}
