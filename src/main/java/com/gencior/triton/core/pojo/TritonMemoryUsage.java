package com.gencior.triton.core.pojo;

import inference.GrpcService;

/**
 * Encapsulates memory usage information for a specific memory device on the Triton server.
 *
 * <p>This class represents memory usage statistics for a particular memory device, such as GPU
 * memory (e.g., NVIDIA GPU VRAM) or CPU system memory. It tracks the total amount of memory
 * currently allocated or reserved on a specific memory device identified by its ID.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code MemoryUsage}.
 *
 * <h2>Memory Types:</h2>
 * <ul>
 *   <li><strong>gpu:</strong> GPU device memory (e.g., CUDA device memory)</li>
 *   <li><strong>cpu:</strong> CPU system memory</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * List<TritonMemoryUsage> memoryUsages = modelStatistics.getMemoryUsage();
 * for (TritonMemoryUsage usage : memoryUsages) {
 *     double memoryMb = usage.getByteSize() / (1024.0 * 1024.0);
 *     System.out.printf("%s device %d: %.2f MB%n", usage.getType(), usage.getId(), memoryMb);
 * }
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonMemoryUsage {
    private final String type;
    private final long id;
    private final long byteSize;

    private TritonMemoryUsage(String type, long id, long byteSize) {
        this.type = type;
        this.id = id;
        this.byteSize = byteSize;
    }

    /**
     * Creates a TritonMemoryUsage from a gRPC MemoryUsage message.
     *
     * @param proto the gRPC MemoryUsage message from Triton server
     * @return a new TritonMemoryUsage instance
     */
    public static TritonMemoryUsage fromProto(GrpcService.MemoryUsage proto) {
        return new TritonMemoryUsage(proto.getType(), proto.getId(), proto.getByteSize());
    }

    /**
     * Returns the type of memory device.
     *
     * <p>Typical values are "gpu" for GPU memory or "cpu" for CPU memory.
     *
     * @return the memory type (e.g., "gpu", "cpu")
     */
    public String getType() { return type; }

    /**
     * Returns the ID of the memory device.
     *
     * <p>For GPUs, this is typically the CUDA device ID (0, 1, 2, etc.).
     * For CPU memory, this is typically 0.
     *
     * @return the device ID
     */
    public long getId() { return id; }

    /**
     * Returns the amount of memory used by this device in bytes.
     *
     * @return the memory size in bytes
     */
    public long getByteSize() { return byteSize; }
}
