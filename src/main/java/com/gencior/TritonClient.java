package com.gencior;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import com.gencior.core.InferInput;
import com.gencior.core.InferResult;
import com.gencior.core.pojo.TritonModelConfig;
import com.gencior.core.pojo.TritonModelMetadata;
import com.gencior.core.pojo.TritonModelStatistics;
import com.gencior.core.pojo.TritonRepositoryIndex;
import com.gencior.core.pojo.TritonServerMetadata;

import inference.GrpcService;

/**
 *
 * @author sachachoumiloff
 */

public interface TritonClient extends AutoCloseable {

    boolean isServerLive();
    boolean isServerReady();
    TritonServerMetadata getServerMetadata();
    boolean isModelReady(String modelId);
    boolean isModelReady(String modelId, String modelVersion);
    TritonModelMetadata getModelMetadata(String modelId, String modelVersion);
    TritonModelConfig getModelConfig(String modelId);
    TritonModelConfig getModelConfig(String modelId, String modelVersion);    
    TritonRepositoryIndex getModelRepositoryIndex();
    void loadModel(String modelId);
    void unLoadModel(String modelId);
    List<TritonModelStatistics> getInferenceStatistics(String modelId, String modelVersion);
    InferResult infer(String modelId, List<InferInput> inputs);
    InferResult infer(String modelId, String modelVersion, List<InferInput> inputs, Map<String, GrpcService.InferParameter> customParameters);
    CompletableFuture<InferResult> inferAsync(String modelId, List<InferInput> inputs);
    CompletableFuture<InferResult> inferAsync(String modelId, String modelVersion, List<InferInput> inputs, Map<String, GrpcService.InferParameter> customParameters);
}
