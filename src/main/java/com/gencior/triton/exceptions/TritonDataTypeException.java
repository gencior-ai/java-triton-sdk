package com.gencior.triton.exceptions;

import java.util.Arrays;

import com.gencior.triton.core.TritonDataType;

/**
 *
 * @author sachachoumiloff
 */
public class TritonDataTypeException extends TritonInferException {
    
    public TritonDataTypeException(TritonDataType actual, TritonDataType[] expected) {
        super(String.format(
            "Datatype mismatch: The input tensor is defined as %s, but you tried to set data of type(s) %s",
            actual.getTritonName(), Arrays.toString(expected)
        ));
    }

    public TritonDataTypeException(String message) {
        super(message);
    }
}
