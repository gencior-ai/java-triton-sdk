package com.gencior.triton;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;

import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferParameters;
import com.gencior.triton.core.InferRequestedOutput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.InferStreamHandle;
import com.gencior.triton.core.InferStreamListener;
import com.gencior.triton.core.pojo.TritonModelConfig;
import com.gencior.triton.core.pojo.TritonModelMetadata;
import com.gencior.triton.core.pojo.TritonModelStatistics;
import com.gencior.triton.core.pojo.TritonRepositoryIndex;
import com.gencior.triton.core.pojo.TritonServerMetadata;

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
    InferResult infer(String modelId, String modelVersion, List<InferInput> inputs, InferParameters customParameters);
    InferResult infer(String modelId, String modelVersion, List<InferInput> inputs, List<InferRequestedOutput> outputs, InferParameters customParameters);
    CompletableFuture<InferResult> inferAsync(String modelId, List<InferInput> inputs);
    CompletableFuture<InferResult> inferAsync(String modelId, String modelVersion, List<InferInput> inputs, InferParameters customParameters);
    CompletableFuture<InferResult> inferAsync(String modelId, String modelVersion, List<InferInput> inputs, List<InferRequestedOutput> outputs, InferParameters customParameters);
    InferStreamHandle inferStream(String modelId, List<InferInput> inputs, InferStreamListener listener);
    InferStreamHandle inferStream(String modelId, String modelVersion, List<InferInput> inputs, InferParameters customParameters, InferStreamListener listener);
    InferStreamHandle inferStream(String modelId, String modelVersion, List<InferInput> inputs, List<InferRequestedOutput> outputs, InferParameters customParameters, InferStreamListener listener);
    Flow.Publisher<InferResult> inferStreamPublisher(String modelId, List<InferInput> inputs);
    Flow.Publisher<InferResult> inferStreamPublisher(String modelId, String modelVersion, List<InferInput> inputs, InferParameters customParameters);
    Flow.Publisher<InferResult> inferStreamPublisher(String modelId, String modelVersion, List<InferInput> inputs, List<InferRequestedOutput> outputs, InferParameters customParameters);
}
