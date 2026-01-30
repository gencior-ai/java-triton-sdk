package com.gencior.triton.http;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import com.gencior.triton.TritonClient;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.pojo.TritonModelConfig;
import com.gencior.triton.core.pojo.TritonModelMetadata;
import com.gencior.triton.core.pojo.TritonModelStatistics;
import com.gencior.triton.core.pojo.TritonRepositoryIndex;
import com.gencior.triton.core.pojo.TritonServerMetadata;

import inference.GrpcService;

/**
 *
 * @author sachachoumiloff
 */
public class TritonHttpClient implements TritonClient {

    @Override
    public void close() throws Exception {
        throw new UnsupportedOperationException("Unimplemented method 'close'");
    }

    @Override
    public boolean isServerLive() {
        throw new UnsupportedOperationException("Unimplemented method 'isServerLive'");
    }

    @Override
    public boolean isServerReady() {
        throw new UnsupportedOperationException("Unimplemented method 'isServerReady'");
    }

    @Override
    public TritonServerMetadata getServerMetadata() {
        throw new UnsupportedOperationException("Unimplemented method 'getServerMetadata'");
    }

    @Override
    public boolean isModelReady(String modelId) {
        throw new UnsupportedOperationException("Unimplemented method 'isModelReady'");
    }

    @Override
    public boolean isModelReady(String modelId, String modelVersion) {
        throw new UnsupportedOperationException("Unimplemented method 'isModelReady'");
    }

    @Override
    public TritonModelMetadata getModelMetadata(String modelId, String modelVersion) {
        throw new UnsupportedOperationException("Unimplemented method 'getModelMetadata'");
    }

    @Override
    public TritonModelConfig getModelConfig(String modelId) {
        throw new UnsupportedOperationException("Unimplemented method 'getModelConfig'");
    }

    @Override
    public TritonModelConfig getModelConfig(String modelId, String modelVersion) {
        throw new UnsupportedOperationException("Unimplemented method 'getModelConfig'");
    }

    @Override
    public TritonRepositoryIndex getModelRepositoryIndex() {
        throw new UnsupportedOperationException("Unimplemented method 'getModelRepositoryIndex'");
    }

    @Override
    public void loadModel(String modelId) {
        throw new UnsupportedOperationException("Unimplemented method 'loadModel'");
    }

    @Override
    public void unLoadModel(String modelId) {
        throw new UnsupportedOperationException("Unimplemented method 'unLoadModel'");
    }

    @Override
    public List<TritonModelStatistics> getInferenceStatistics(String modelId, String modelVersion) {
        throw new UnsupportedOperationException("Unimplemented method 'getInferenceStatistics'");
    }

    @Override
    public InferResult infer(String modelId, List<InferInput> inputs) {
        throw new UnsupportedOperationException("Unimplemented method 'infer'");
    }

    @Override
    public InferResult infer(String modelId, String modelVersion, List<InferInput> inputs,
            Map<String, GrpcService.InferParameter> customParameters) {
        throw new UnsupportedOperationException("Unimplemented method 'infer'");
    }

    @Override
    public CompletableFuture<InferResult> inferAsync(String modelId, List<InferInput> inputs) {
        throw new UnsupportedOperationException("Unimplemented method 'inferAsync'");
    }

    @Override
    public CompletableFuture<InferResult> inferAsync(String modelId, String modelVersion, List<InferInput> inputs,
            Map<String, GrpcService.InferParameter> customParameters) {
        throw new UnsupportedOperationException("Unimplemented method 'inferAsync'");
    }

}
