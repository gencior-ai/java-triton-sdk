package com.gencior.triton.core.pojo;

import com.fasterxml.jackson.databind.JsonNode;

import inference.GrpcService;

/**
 * Encapsulates comprehensive inference statistics for a model.
 *
 * <p>This class aggregates timing and success/failure metrics for all inference requests,
 * including queue times, compute times, and cache hit/miss statistics. It provides detailed
 * insights into model performance and inference processing stages.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code InferStatistics}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonInferStatistics {
    private final TritonStatisticDuration success;
    private final TritonStatisticDuration fail;
    private final TritonStatisticDuration queue;
    private final TritonStatisticDuration computeInput;
    private final TritonStatisticDuration computeInfer;
    private final TritonStatisticDuration computeOutput;
    private final TritonStatisticDuration cacheHit;
    private final TritonStatisticDuration cacheMiss;

    private TritonInferStatistics(TritonStatisticDuration success, TritonStatisticDuration fail, 
                                 TritonStatisticDuration queue, TritonStatisticDuration computeInput, 
                                 TritonStatisticDuration computeInfer, TritonStatisticDuration computeOutput, 
                                 TritonStatisticDuration cacheHit, TritonStatisticDuration cacheMiss) {
        this.success = success;
        this.fail = fail;
        this.queue = queue;
        this.computeInput = computeInput;
        this.computeInfer = computeInfer;
        this.computeOutput = computeOutput;
        this.cacheHit = cacheHit;
        this.cacheMiss = cacheMiss;
    }

    public static TritonInferStatistics fromProto(GrpcService.InferStatistics proto) {
        return new TritonInferStatistics(
            TritonStatisticDuration.fromProto(proto.getSuccess()),
            TritonStatisticDuration.fromProto(proto.getFail()),
            TritonStatisticDuration.fromProto(proto.getQueue()),
            TritonStatisticDuration.fromProto(proto.getComputeInput()),
            TritonStatisticDuration.fromProto(proto.getComputeInfer()),
            TritonStatisticDuration.fromProto(proto.getComputeOutput()),
            TritonStatisticDuration.fromProto(proto.getCacheHit()),
            TritonStatisticDuration.fromProto(proto.getCacheMiss())
        );
    }

    /**
     * Creates a TritonInferStatistics from a JSON response.
     *
     * @param json the JSON node containing duration fields such as {@code success},
     *             {@code fail}, {@code queue}, {@code compute_input}, {@code compute_infer},
     *             {@code compute_output}, {@code cache_hit}, and {@code cache_miss}
     * @return a new TritonInferStatistics instance
     */
    public static TritonInferStatistics fromJson(JsonNode json) {
        return new TritonInferStatistics(
                TritonStatisticDuration.fromJson(json.path("success")),
                TritonStatisticDuration.fromJson(json.path("fail")),
                TritonStatisticDuration.fromJson(json.path("queue")),
                TritonStatisticDuration.fromJson(json.path("compute_input")),
                TritonStatisticDuration.fromJson(json.path("compute_infer")),
                TritonStatisticDuration.fromJson(json.path("compute_output")),
                TritonStatisticDuration.fromJson(json.path("cache_hit")),
                TritonStatisticDuration.fromJson(json.path("cache_miss"))
        );
    }

    /** Returns statistics for successfully completed inferences. */
    public TritonStatisticDuration getSuccess () {return this.success;}

    /** Returns statistics for failed inferences. */
    public TritonStatisticDuration getFail () {return this.fail;}

    /** Returns statistics for time spent in the request queue. */
    public TritonStatisticDuration getQueue () {return this.queue;}

    /** Returns statistics for time spent preparing input tensors. */
    public TritonStatisticDuration getComputeInput() {return this.computeInput;}

    /** Returns statistics for time spent in model inference computation. */
    public TritonStatisticDuration getComputeOutput () {return this.computeOutput;}

    /** Returns statistics for time spent in inference computation. */
    public TritonStatisticDuration getComputeInfer () {return this.computeInfer;}

    /** Returns statistics for inferences that hit the result cache. */
    public TritonStatisticDuration getCacheHit () {return this.cacheHit;}

    /** Returns statistics for inferences that missed the result cache. */
    public TritonStatisticDuration getCacheMiss () {return this.cacheMiss;}
}
