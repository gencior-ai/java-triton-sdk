package com.gencior.triton.core.pojo;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Tests for the fromJson() factory methods on all POJOs.
 * Verifies that JSON deserialization produces the same results as fromProto().
 */
public class TritonPojoJsonTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    public void testServerMetadataFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "triton", "version": "2.42.0", "extensions": ["classification", "sequence", "model_repository"]}
                """);
        TritonServerMetadata meta = TritonServerMetadata.fromJson(json);
        assertEquals("triton", meta.getName());
        assertEquals("2.42.0", meta.getVersion());
        assertEquals(3, meta.getExtensions().size());
        assertEquals("classification", meta.getExtensions().get(0));
        assertTrue(meta.supportsExtension("sequence"));
    }

    @Test
    public void testServerMetadataFromJsonEmptyExtensions() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "triton", "version": "2.42.0"}
                """);
        TritonServerMetadata meta = TritonServerMetadata.fromJson(json);
        assertEquals("triton", meta.getName());
        assertTrue(meta.getExtensions().isEmpty());
    }

    @Test
    public void testTensorMetadataFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "input_0", "datatype": "FP32", "shape": [1, 3, 224, 224]}
                """);
        TritonTensorMetadata tensor = TritonTensorMetadata.fromJson(json);
        assertEquals("input_0", tensor.getName());
        assertEquals("FP32", tensor.getDatatype());
        assertEquals(List.of(1L, 3L, 224L, 224L), tensor.getShape());
    }

    @Test
    public void testTensorMetadataFromJsonDynamicShape() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "text", "datatype": "BYTES", "shape": [-1]}
                """);
        TritonTensorMetadata tensor = TritonTensorMetadata.fromJson(json);
        assertEquals("BYTES", tensor.getDatatype());
        assertEquals(List.of(-1L), tensor.getShape());
    }


    @Test
    public void testModelMetadataFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {
                    "name": "inception_v3",
                    "versions": ["1", "2"],
                    "platform": "tensorflow_savedmodel",
                    "inputs": [
                        {"name": "input", "datatype": "FP32", "shape": [1, 299, 299, 3]}
                    ],
                    "outputs": [
                        {"name": "output", "datatype": "FP32", "shape": [1, 1001]}
                    ]
                }
                """);
        TritonModelMetadata meta = TritonModelMetadata.fromJson(json);
        assertEquals("inception_v3", meta.getName());
        assertEquals(List.of("1", "2"), meta.getVersions());
        assertEquals("tensorflow_savedmodel", meta.getPlatform());
        assertEquals(1, meta.getInputs().size());
        assertEquals("input", meta.getInputs().get(0).getName());
        assertEquals("FP32", meta.getInputs().get(0).getDatatype());
        assertEquals(1, meta.getOutputs().size());
        assertEquals("output", meta.getOutputs().get(0).getName());
    }

    @Test
    public void testModelMetadataFromJsonNoInputsOutputs() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "empty_model", "platform": "python"}
                """);
        TritonModelMetadata meta = TritonModelMetadata.fromJson(json);
        assertEquals("empty_model", meta.getName());
        assertTrue(meta.getInputs().isEmpty());
        assertTrue(meta.getOutputs().isEmpty());
        assertTrue(meta.getVersions().isEmpty());
    }


    @Test
    public void testModelConfigFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {
                    "name": "resnet50",
                    "platform": "",
                    "backend": "python",
                    "runtime": "",
                    "max_batch_size": 8,
                    "default_model_filename": "model.py",
                    "cc_model_filenames": {"75": "model_sm75.plan"},
                    "metric_tags": {"env": "prod"}
                }
                """);
        TritonModelConfig config = TritonModelConfig.fromJson(json);
        assertEquals("resnet50", config.getName());
        assertEquals("python", config.getBackend());
        assertEquals(8, config.getMaxBatchSize());
        assertEquals("model.py", config.getDefaultModelFilename());
        assertEquals(Map.of("75", "model_sm75.plan"), config.getCcModelFilenames());
        assertEquals(Map.of("env", "prod"), config.getMetricTags());
    }

    @Test
    public void testModelConfigFromJsonMinimal() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "simple", "backend": "onnxruntime"}
                """);
        TritonModelConfig config = TritonModelConfig.fromJson(json);
        assertEquals("simple", config.getName());
        assertEquals("onnxruntime", config.getBackend());
        assertEquals(0, config.getMaxBatchSize());
        assertTrue(config.getCcModelFilenames().isEmpty());
        assertTrue(config.getMetricTags().isEmpty());
    }


    @Test
    public void testModelIndexFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "bert", "version": "1", "state": "READY", "reason": ""}
                """);
        TritonModelIndex index = TritonModelIndex.fromJson(json);
        assertEquals("bert", index.getName());
        assertEquals("1", index.getVersion());
        assertEquals("READY", index.getState());
        assertEquals("", index.getReason());
    }

    @Test
    public void testModelIndexFromJsonUnavailable() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "broken", "version": "1", "state": "UNAVAILABLE", "reason": "model file not found"}
                """);
        TritonModelIndex index = TritonModelIndex.fromJson(json);
        assertEquals("UNAVAILABLE", index.getState());
        assertEquals("model file not found", index.getReason());
    }


    @Test
    public void testRepositoryIndexFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                [
                    {"name": "model_a", "version": "1", "state": "READY", "reason": ""},
                    {"name": "model_b", "version": "2", "state": "LOADING", "reason": ""}
                ]
                """);
        TritonRepositoryIndex repoIndex = TritonRepositoryIndex.fromJson(json);
        assertEquals(2, repoIndex.getModels().size());
        assertEquals("model_a", repoIndex.getModels().get(0).getName());
        assertEquals("model_b", repoIndex.getModels().get(1).getName());
        assertEquals("LOADING", repoIndex.getModels().get(1).getState());
    }

    @Test
    public void testRepositoryIndexFromJsonEmpty() throws Exception {
        JsonNode json = MAPPER.readTree("[]");
        TritonRepositoryIndex repoIndex = TritonRepositoryIndex.fromJson(json);
        assertTrue(repoIndex.getModels().isEmpty());
    }


    @Test
    public void testStatisticDurationFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"count": 1500, "ns": 3000000000}
                """);
        TritonStatisticDuration duration = TritonStatisticDuration.fromJson(json);
        assertEquals(1500, duration.getCount());
        assertEquals(3000000000L, duration.getTotalTimeNs());
        assertEquals(3000.0, duration.getTotalTimeMs(), 0.001);
    }

    @Test
    public void testStatisticDurationFromJsonDefaults() throws Exception {
        JsonNode json = MAPPER.readTree("{}");
        TritonStatisticDuration duration = TritonStatisticDuration.fromJson(json);
        assertEquals(0, duration.getCount());
        assertEquals(0, duration.getTotalTimeNs());
    }


    @Test
    public void testMemoryUsageFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"type": "GPU", "id": 0, "byte_size": 1073741824}
                """);
        TritonMemoryUsage usage = TritonMemoryUsage.fromJson(json);
        assertEquals("GPU", usage.getType());
        assertEquals(0, usage.getId());
        assertEquals(1073741824L, usage.getByteSize());
    }


    @Test
    public void testInferStatisticsFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {
                    "success": {"count": 100, "ns": 5000000},
                    "fail": {"count": 2, "ns": 100000},
                    "queue": {"count": 102, "ns": 200000},
                    "compute_input": {"count": 100, "ns": 300000},
                    "compute_infer": {"count": 100, "ns": 4000000},
                    "compute_output": {"count": 100, "ns": 500000},
                    "cache_hit": {"count": 10, "ns": 50000},
                    "cache_miss": {"count": 90, "ns": 4500000}
                }
                """);
        TritonInferStatistics stats = TritonInferStatistics.fromJson(json);
        assertEquals(100, stats.getSuccess().getCount());
        assertEquals(2, stats.getFail().getCount());
        assertEquals(102, stats.getQueue().getCount());
        assertEquals(100, stats.getComputeInput().getCount());
        assertEquals(100, stats.getComputeInfer().getCount());
        assertEquals(100, stats.getComputeOutput().getCount());
        assertEquals(10, stats.getCacheHit().getCount());
        assertEquals(90, stats.getCacheMiss().getCount());
    }


    @Test
    public void testInferResponseStatisticsFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {
                    "compute_infer": {"count": 50, "ns": 2000000},
                    "compute_output": {"count": 50, "ns": 1000000},
                    "success": {"count": 48, "ns": 3000000},
                    "fail": {"count": 2, "ns": 50000},
                    "empty_response": {"count": 0, "ns": 0},
                    "cancel": {"count": 1, "ns": 10000}
                }
                """);
        TritonInferResponseStatistics stats = TritonInferResponseStatistics.fromJson(json);
        assertEquals(50, stats.getComputeInfer().getCount());
        assertEquals(50, stats.getComputeOutput().getCount());
        assertEquals(48, stats.getSuccess().getCount());
        assertEquals(2, stats.getFail().getCount());
        assertEquals(0, stats.getEmptyResponse().getCount());
        assertEquals(1, stats.getCancel().getCount());
    }


    @Test
    public void testModelStatisticsFromJson() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {
                    "name": "resnet50",
                    "version": "1",
                    "last_inference": 1700000000,
                    "inference_count": 500,
                    "execution_count": 100,
                    "inference_stats": {
                        "success": {"count": 500, "ns": 50000000},
                        "fail": {"count": 0, "ns": 0},
                        "queue": {"count": 500, "ns": 10000000},
                        "compute_input": {"count": 500, "ns": 5000000},
                        "compute_infer": {"count": 500, "ns": 30000000},
                        "compute_output": {"count": 500, "ns": 5000000},
                        "cache_hit": {"count": 0, "ns": 0},
                        "cache_miss": {"count": 0, "ns": 0}
                    },
                    "memory_usage": [
                        {"type": "GPU", "id": 0, "byte_size": 536870912}
                    ],
                    "response_stats": {}
                }
                """);
        TritonModelStatistics stats = TritonModelStatistics.fromJson(json);
        assertEquals("resnet50", stats.getName());
        assertEquals("1", stats.getVersion());
        assertEquals(1700000000L, stats.getLastInference());
        assertEquals(500, stats.getInferenceCount());
        assertEquals(100, stats.getExecustionCount());
        assertEquals(500, stats.getTritonInferStatistics().getSuccess().getCount());
        assertEquals(5.0, stats.getBatchingEfficiency(), 0.001);
        assertEquals(1.0, stats.getSuccessRate(), 0.001);
        assertEquals(536870912L, stats.getTotalGpuMemoryUsage());
    }


    @Test(expected = UnsupportedOperationException.class)
    public void testServerMetadataExtensionsImmutable() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "t", "version": "1", "extensions": ["a"]}
                """);
        TritonServerMetadata meta = TritonServerMetadata.fromJson(json);
        meta.getExtensions().add("should_fail");
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testModelMetadataVersionsImmutable() throws Exception {
        JsonNode json = MAPPER.readTree("""
                {"name": "m", "versions": ["1"], "platform": "p"}
                """);
        TritonModelMetadata meta = TritonModelMetadata.fromJson(json);
        meta.getVersions().add("should_fail");
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testRepositoryIndexModelsImmutable() throws Exception {
        JsonNode json = MAPPER.readTree("""
                [{"name": "m", "version": "1", "state": "READY", "reason": ""}]
                """);
        TritonRepositoryIndex index = TritonRepositoryIndex.fromJson(json);
        index.getModels().add(null);
    }
}
