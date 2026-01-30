package com.gencior.triton.core.pojo;

import inference.GrpcService;

/**
 * Encapsulates metadata about a single model in the Triton repository index.
 *
 * <p>This class represents a model entry from the repository index, providing essential metadata
 * about a model's availability and health status. Each model index entry indicates whether the model
 * is currently ready for inference or if it's unavailable and why.
 *
 * <p>This is an immutable object that wraps the gRPC message
 * {@code RepositoryIndexResponse.ModelIndex}.
 *
 * <h2>State Values:</h2>
 * <ul>
 *   <li><strong>READY:</strong> Model is loaded and ready to accept inference requests</li>
 *   <li><strong>UNAVAILABLE:</strong> Model is not ready; check {@link #getReason()} for details</li>
 *   <li><strong>LOADING:</strong> Model is currently being loaded (transient state)</li>
 *   <li><strong>UNLOADING:</strong> Model is currently being unloaded (transient state)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * List<TritonModelIndex> models = repositoryIndex.getModels();
 * for (TritonModelIndex model : models) {
 *     System.out.println("Model: " + model.getName() + " v" + model.getVersion());
 *     if (!"READY".equals(model.getState())) {
 *         System.out.println("Status: " + model.getState());
 *         System.out.println("Reason: " + model.getReason());
 *     }
 * }
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonModelIndex {

    private final String name;
    private final String version;
    private final String state;
    private final String reason;

    private TritonModelIndex(String name, String version, String state, String reason) {
        this.name = name;
        this.version = version;
        this.state = state;
        this.reason = reason;
    }

    /**
     * Creates a TritonModelIndex from a gRPC ModelIndex message.
     *
     * @param proto the gRPC ModelIndex message from Triton server
     * @return a new TritonModelIndex instance
     */
    public static TritonModelIndex fromProto(GrpcService.RepositoryIndexResponse.ModelIndex proto) {
        return new TritonModelIndex(
                proto.getName(),
                proto.getVersion(),
                proto.getState(),
                proto.getReason()
        );
    }

    /**
     * Returns the name of the model.
     *
     * @return the model name as it appears in the repository
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the version of the model.
     *
     * @return the model version (typically a semantic version like "1.0" or similar identifier)
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns the current state of the model.
     *
     * @return the state (typically "READY", "UNAVAILABLE", "LOADING", or "UNLOADING")
     */
    public String getState() {
        return state;
    }

    /**
     * Returns the reason for the current state, if not ready.
     *
     * <p>This field is typically empty when the model state is "READY", and contains
     * an error message or explanation when the model is unavailable.
     *
     * @return the reason text explaining the state, empty if not applicable
     */
    public String getReason() {
        return reason;
    }

    @Override
    public String toString() {
        return String.format("ModelIndex[name=%s, version=%s, state=%s]", name, version, state);
    }
}
