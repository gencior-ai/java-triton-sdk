package com.gencior.triton.core.pojo;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import inference.GrpcService;

/**
 * Encapsulates the repository index information from a Triton server.
 *
 * <p>This class provides an overview of all models available in the server's model repository.
 * It contains a list of model index entries, each describing a model's location and basic
 * metadata.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code RepositoryIndexResponse}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonRepositoryIndex {

    private final List<TritonModelIndex> models;

    private TritonRepositoryIndex(List<TritonModelIndex> models) {
        this.models = models != null ? List.copyOf(models) : Collections.emptyList();
    }

    /**
     * Creates a TritonRepositoryIndex from a gRPC RepositoryIndexResponse message.
     *
     * @param response the gRPC response message from Triton server
     * @return a new TritonRepositoryIndex instance
     */
    public static TritonRepositoryIndex fromProto(GrpcService.RepositoryIndexResponse response) {
        List<TritonModelIndex> modelList = response.getModelsList().stream()
                .map(TritonModelIndex::fromProto)
                .collect(Collectors.toList());
        
        return new TritonRepositoryIndex(modelList);
    }

    public List<TritonModelIndex> getModels() {
        return models;
    }

    @Override
    public String toString() {
        return "TritonRepositoryIndex{modelsCount=" + models.size() + "}";
    }
}