package com.gencior.exceptions;

import java.util.Arrays;

/**
 *
 * @author sachachoumiloff
 */
public class TritonShapeMismatchException extends TritonInferException {
    public TritonShapeMismatchException(long[] shape, int gotElements, long expectedElements) {
        super(String.format(
            "Dimension mismatch: The shape %s expects %d elements, but you provided a data array of size %d",
            Arrays.toString(shape), expectedElements, gotElements
        ));
    }
}