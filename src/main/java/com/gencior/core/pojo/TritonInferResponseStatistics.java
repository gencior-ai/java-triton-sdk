package com.gencior.core.pojo;

import inference.GrpcService;

/**
 * Encapsulates statistics for inference response times.
 *
 * <p>This class represents statistics about the timing of various stages in an inference response,
 * including compute time, output processing time, and request status distribution. These metrics
 * are obtained from the Triton Inference Server gRPC API.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code InferResponseStatistics}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonInferResponseStatistics {

    private final TritonStatisticDuration computeInfer;
    private final TritonStatisticDuration computeOutput;
    private final TritonStatisticDuration success;
    private final TritonStatisticDuration fail;
    private final TritonStatisticDuration emptyResponse;
    private final TritonStatisticDuration cancel;

    private TritonInferResponseStatistics(TritonStatisticDuration computeInfer,
            TritonStatisticDuration computeOutput,
            TritonStatisticDuration success,
            TritonStatisticDuration fail,
            TritonStatisticDuration emptyResponse,
            TritonStatisticDuration cancel) {
        this.computeInfer = computeInfer;
        this.computeOutput = computeOutput;
        this.success = success;
        this.fail = fail;
        this.emptyResponse = emptyResponse;
        this.cancel = cancel;
    }

    public static TritonInferResponseStatistics fromProto(GrpcService.InferResponseStatistics proto) {
        return new TritonInferResponseStatistics(
                TritonStatisticDuration.fromProto(proto.getComputeInfer()),
                TritonStatisticDuration.fromProto(proto.getComputeOutput()),
                TritonStatisticDuration.fromProto(proto.getSuccess()),
                TritonStatisticDuration.fromProto(proto.getFail()),
                TritonStatisticDuration.fromProto(proto.getEmptyResponse()),
                TritonStatisticDuration.fromProto(proto.getCancel())
        );
    }

    /**
     * Returns statistics for the time spent in model inference computation.
     *
     * @return duration statistics for inference computation
     */
    public TritonStatisticDuration getComputeInfer() {
        return computeInfer;
    }

    /**
     * Returns statistics for the time spent preparing output tensors.
     *
     * @return duration statistics for output preparation
     */
    public TritonStatisticDuration getComputeOutput() {
        return computeOutput;
    }

    /**
     * Returns statistics for successfully completed inference requests.
     *
     * @return duration statistics for successful requests
     */
    public TritonStatisticDuration getSuccess() {
        return success;
    }

    /**
     * Returns statistics for inference requests that failed.
     *
     * @return duration statistics for failed requests
     */
    public TritonStatisticDuration getFail() {
        return fail;
    }

    /**
     * Returns statistics for inference requests with empty responses.
     *
     * @return duration statistics for empty responses
     */
    public TritonStatisticDuration getEmptyResponse() {
        return emptyResponse;
    }

    /**
     * Returns statistics for cancelled inference requests.
     *
     * @return duration statistics for cancelled requests
     */
    public TritonStatisticDuration getCancel() {
        return cancel;
    }

    @Override
    public String toString() {
        return String.format("ResponseStats[success=%d, fail=%d]",
                success.getCount(), fail.getCount());
    }
}
