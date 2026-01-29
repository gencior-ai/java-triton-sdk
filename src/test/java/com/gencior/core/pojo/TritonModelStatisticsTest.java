package com.gencior.core.pojo;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.gencior.core.pojo.TritonModelStatistics;

import inference.GrpcService.InferStatistics;
import inference.GrpcService.ModelStatistics;
import inference.GrpcService.StatisticDuration;

public class TritonModelStatisticsTest {

    @Test
    public void testStatisticsCalculations_Success() {
        StatisticDuration computeProto = StatisticDuration.newBuilder()
                .setCount(10)
                .setNs(50_000_000L)
                .build();
        StatisticDuration successProto = StatisticDuration.newBuilder().setCount(8).build();
        StatisticDuration failProto = StatisticDuration.newBuilder().setCount(2).build();

        InferStatistics inferStatsProto = InferStatistics.newBuilder()
                .setComputeInfer(computeProto)
                .setSuccess(successProto)
                .setFail(failProto)
                .setQueue(StatisticDuration.newBuilder().setCount(10).setNs(10_000_000L).build()) // 1ms avg queue
                .build();

        ModelStatistics modelStatsProto = ModelStatistics.newBuilder()
                .setName("resnet50")
                .setVersion("1")
                .setInferenceCount(20)
                .setExecutionCount(5)
                .setInferenceStats(inferStatsProto)
                .build();

        TritonModelStatistics stats = TritonModelStatistics.fromProto(modelStatsProto);

        assertEquals("resnet50", stats.getName());
        assertEquals("1", stats.getVersion());
        assertEquals("Average Compute Ms", 5.0, stats.getAverageComputeMs(), 0.001);
        assertEquals("Average Queue Ms", 1.0, stats.getAverageQueueMs(), 0.001);
        assertEquals("Success Rate", 0.8, stats.getSuccessRate(), 0.001);
        assertEquals("Batching Efficiency", 4.0, stats.getBatchingEfficiency(), 0.001);
        assertEquals("resnet50 (v1)", stats.getModelIdentifier());
    }

    @Test
    public void testStatistics_ZeroData_NoExceptions() {
        ModelStatistics emptyProto = ModelStatistics.newBuilder()
                .setName("idle_model")
                .setInferenceCount(0)
                .setExecutionCount(0)
                .setInferenceStats(InferStatistics.newBuilder().build())
                .build();

        TritonModelStatistics stats = TritonModelStatistics.fromProto(emptyProto);

        assertEquals(0.0, stats.getAverageComputeMs(), 0.001);
        assertEquals(0.0, stats.getAverageQueueMs(), 0.001);
        assertEquals(0.0, stats.getBatchingEfficiency(), 0.001);
        assertEquals(1.0, stats.getSuccessRate(), 0.001);
    }
}
