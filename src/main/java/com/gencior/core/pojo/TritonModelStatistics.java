package com.gencior.core.pojo;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import inference.GrpcService;

/**
 * Encapsulates comprehensive statistics for a deployed Triton model.
 *
 * <p>This class aggregates performance metrics for a specific model including inference counts,
 * timing statistics, memory usage, and response statistics. It provides convenient methods to
 * calculate derived metrics such as average latency, success rate, and memory usage.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code ModelStatistics}.
 *
 * <h2>Convenience Methods:</h2>
 * <ul>
 *   <li>{@link #getAverageComputeMs()} - Calculate average inference computation time</li>
 *   <li>{@link #getSuccessRate()} - Calculate percentage of successful inferences</li>
 *   <li>{@link #getBatchingEfficiency()} - Calculate batching efficiency ratio</li>
 *   <li>{@link #getTotalGpuMemoryUsage()} - Sum GPU memory usage across all instances</li>
 * </ul>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonModelStatistics {

    private final String name;
    private final String version;
    private final long lastInference;
    private final long inferenceCount;
    private final long executionCount;
    private final TritonInferStatistics inferenceStats;
    private final List<TritonMemoryUsage> memoryUsage;
    private final Map<String, TritonInferResponseStatistics> responseStats;

    private TritonModelStatistics(String name, String version, long lastInference,
            long inferenceCount, long executionCount,
            TritonInferStatistics inferenceStats,
            List<TritonMemoryUsage> memoryUsage,
            Map<String, TritonInferResponseStatistics> responseStats) {
        this.name = name;
        this.version = version;
        this.lastInference = lastInference;
        this.inferenceCount = inferenceCount;
        this.executionCount = executionCount;
        this.inferenceStats = inferenceStats;
        this.memoryUsage = List.copyOf(memoryUsage);
        this.responseStats = Map.copyOf(responseStats);
    }

    public static TritonModelStatistics fromProto(GrpcService.ModelStatistics proto) {
        Map<String, TritonInferResponseStatistics> responseMap = proto.getResponseStatsMap().entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> TritonInferResponseStatistics.fromProto(e.getValue())
                ));

        return new TritonModelStatistics(
                proto.getName(),
                proto.getVersion(),
                proto.getLastInference(),
                proto.getInferenceCount(),
                proto.getExecutionCount(),
                TritonInferStatistics.fromProto(proto.getInferenceStats()),
                proto.getMemoryUsageList().stream().map(TritonMemoryUsage::fromProto).collect(Collectors.toList()),
                responseMap
        );
    }

    /**
     * Returns the average inference computation time in milliseconds.
     *
     * <p>This is calculated as total compute time divided by the number of inferences.
     *
     * @return the average computation time in milliseconds, or 0.0 if no inferences
     */
    public double getAverageComputeMs() {
        long count = inferenceStats.getComputeInfer().getCount();
        if (count == 0) {
            return 0.0;
        }
        return inferenceStats.getComputeInfer().getTotalTimeMs() / count;
    }

    /**
     * Returns the average queue wait time in milliseconds.
     *
     * <p>This represents the average time requests spent waiting before being processed.
     *
     * @return the average queue wait time in milliseconds, or 0.0 if no inferences
     */
    public double getAverageQueueMs() {
        long count = inferenceStats.getQueue().getCount();
        if (count == 0) {
            return 0.0;
        }
        return inferenceStats.getQueue().getTotalTimeMs() / count;
    }

    /**
     * Returns the success rate as a ratio between 0.0 and 1.0.
     *
     * <p>This is calculated as successful inferences divided by total inferences (successful + failed).
     *
     * @return the success rate (0.0 = all failed, 1.0 = all successful)
     */
    public double getSuccessRate() {
        long total = inferenceStats.getSuccess().getCount() + inferenceStats.getFail().getCount();
        if (total == 0) {
            return 1.0;
        }
        return (double) inferenceStats.getSuccess().getCount() / total;
    }

    /**
     * Returns the batching efficiency ratio.
     *
     * <p>This is calculated as total inferences divided by total execution batches.
     * A higher ratio indicates better batching efficiency.
     *
     * @return the batching efficiency ratio (inferences per execution), or 0.0 if no executions
     */
    public double getBatchingEfficiency() {
        if (executionCount == 0) {
            return 0.0;
        }
        return (double) inferenceCount / executionCount;
    }

    /**
     * Returns the total GPU memory usage in bytes across all model instances.
     *
     * @return total GPU memory usage in bytes
     */
    public long getTotalGpuMemoryUsage() {
        return memoryUsage.stream()
                .filter(m -> "GPU".equalsIgnoreCase(m.getType()))
                .mapToLong(TritonMemoryUsage::getByteSize)
                .sum();
    }

    /**
     * Checks if there have been any cache hits for this model.
     *
     * @return {@code true} if at least one cache hit occurred, {@code false} otherwise
     */
    public boolean hasCacheHits() {
        return inferenceStats.getCacheHit().getCount() > 0;
    }

    /**
     * Returns a human-readable identifier for this model.
     *
     * <p>Format: "model_name (vX.Y.Z)"
     *
     * @return the model identifier string
     */
    public String getModelIdentifier() {
        return String.format("%s (v%s)", name, version);
    }

    /**
     * Returns the name of the model.
     *
     * @return the model name
     */
    public String getName() {
        return this.name;
    }

    /**
     * Returns the total number of inferences executed by this model.
     *
     * @return the total inference count
     */
    public long getInferenceCount() {
        return this.inferenceCount;
    }

    /**
     * Returns the version of the model.
     *
     * @return the model version string
     */
    public String getVersion() {
        return this.version;
    }

    /**
     * Returns the timestamp of the last inference in microseconds (Unix epoch).
     *
     * @return the last inference timestamp
     */
    public long getLastInference() {
        return this.lastInference;
    }

    /**
     * Returns the total number of execution batches for this model.
     *
     * @return the execution count
     */
    public long getExecustionCount() {
        return this.executionCount;
    }

    /**
     * Returns detailed inference timing statistics.
     *
     * @return the inference statistics object containing compute, queue, and status timings
     */
    public TritonInferStatistics getTritonInferStatistics() {
        return this.inferenceStats;
    }
}
