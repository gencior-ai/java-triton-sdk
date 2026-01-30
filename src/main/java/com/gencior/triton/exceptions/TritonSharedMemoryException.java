package com.gencior.triton.exceptions;

/**
 *
 * @author sachachoumiloff
 */
public class TritonSharedMemoryException extends TritonInferException {
    public TritonSharedMemoryException(String message) {
        super(message);
    }

    public static TritonSharedMemoryException dataAccessWhileShared() {
        return new TritonSharedMemoryException(
            "Cannot access raw content because this input is configured to use Shared Memory."
        );
    }
}
