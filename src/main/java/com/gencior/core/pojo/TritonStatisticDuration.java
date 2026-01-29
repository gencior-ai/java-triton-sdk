package com.gencior.core.pojo;

import inference.GrpcService;

/**
 * Encapsulates duration statistics for server operations.
 *
 * <p>This class tracks how many times an operation occurred and the total accumulated time.
 * This is commonly used to track inference timing metrics such as compute time, queue wait time,
 * or input preparation time.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code StatisticDuration}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonStatisticDuration {
    private final long count;
    private final long totalTimeNs;

    private TritonStatisticDuration(long count, long totalTimeNs) {
        this.count = count;
        this.totalTimeNs = totalTimeNs;
    }

    public static TritonStatisticDuration fromProto(GrpcService.StatisticDuration proto) {
        return new TritonStatisticDuration(proto.getCount(), proto.getNs());
    }

    /**
     * Returns the number of times this operation occurred.
     *
     * @return the operation count
     */
    public long getCount() { return count; }

    /**
     * Returns the total accumulated time for all operations in nanoseconds.
     *
     * @return the total time in nanoseconds
     */
    public long getTotalTimeNs() { return totalTimeNs; }

    /**
     * Returns the total accumulated time for all operations in milliseconds.
     *
     * @return the total time in milliseconds as a double
     */
    public double getTotalTimeMs() { return totalTimeNs / 1_000_000.0; }
}
