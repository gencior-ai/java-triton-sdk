package com.gencior.triton.core;

/**
 * Enumeration of data types supported by Triton Inference Server.
 *
 * <p>This enum represents all the data types that can be used for model inputs and outputs when
 * communicating with Triton Inference Server. Each type corresponds to a standard data type
 * recognized by Triton.
 *
 * <h2>Supported Types:</h2>
 * <ul>
 *   <li><strong>Boolean:</strong> BOOL - Boolean value</li>
 *   <li><strong>Unsigned Integers:</strong> UINT8, UINT16, UINT32, UINT64</li>
 *   <li><strong>Signed Integers:</strong> INT8, INT16, INT32, INT64</li>
 *   <li><strong>Floating Point:</strong> FP16 (half precision), FP32 (single precision), FP64 (double precision)</li>
 *   <li><strong>Brain Float:</strong> BF16 (Brain Floating Point)</li>
 *   <li><strong>Variable Length:</strong> BYTES - Variable length byte sequences</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Get data type from string
 * TritonDataType dtype = TritonDataType.fromString("FP32");
 *
 * // Get Triton name representation
 * String tritonName = dtype.getTritonName(); // Returns "FP32"
 * }</pre>
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public enum TritonDataType {
    /** Boolean type: single bit value (true/false) */
    BOOL("BOOL"),

    /** Unsigned 8-bit integer (0 to 255) */
    UINT8("UINT8"),

    /** Unsigned 16-bit integer (0 to 65,535) */
    UINT16("UINT16"),

    /** Unsigned 32-bit integer (0 to 4,294,967,295) */
    UINT32("UINT32"),

    /** Unsigned 64-bit integer (0 to 18,446,744,073,709,551,615) */
    UINT64("UINT64"),

    /** Signed 8-bit integer (-128 to 127) */
    INT8("INT8"),

    /** Signed 16-bit integer (-32,768 to 32,767) */
    INT16("INT16"),

    /** Signed 32-bit integer (-2,147,483,648 to 2,147,483,647) */
    INT32("INT32"),

    /** Signed 64-bit integer (-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807) */
    INT64("INT64"),

    /** 16-bit floating point (half precision) */
    FP16("FP16"),

    /** 32-bit floating point (single precision) */
    FP32("FP32"),

    /** 64-bit floating point (double precision) */
    FP64("FP64"),

    /** Brain Floating Point 16-bit format */
    BF16("BF16"),

    /** Variable length byte sequence */
    BYTES("BYTES");

    private final String tritonName;

    /**
     * Constructs a TritonDataType with the given Triton name.
     *
     * @param tritonName the name of the data type as recognized by Triton Inference Server
     */
    TritonDataType(String tritonName) {
        this.tritonName = tritonName;
    }

    /**
     * Returns the Triton Inference Server representation of this data type.
     *
     * @return the data type name as used by Triton (e.g., "FP32", "INT64")
     */
    public String getTritonName() {
        return tritonName;
    }

    /**
     * Parses a string value and returns the corresponding TritonDataType enum constant.
     *
     * <p>The string value should match one of the Triton data type names (case-sensitive).
     * For example: "FP32", "INT64", "BOOL", "BYTES", etc.
     *
     * @param value the string representation of the data type (must not be null or empty)
     * @return the corresponding TritonDataType enum constant
     * @throws IllegalArgumentException if the value is null or empty
     * @throws IllegalArgumentException if the value does not match any known TritonDataType
     */
    public static TritonDataType fromString(String value) {
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("Value cannot be null or empty");
        }
        for (TritonDataType type : TritonDataType.values()) {
            if (type.tritonName.equals(value)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown TritonDataType: " + value);
    }
}
