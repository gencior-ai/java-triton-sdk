package com.gencior.triton.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

public class InferInputTest {

    private InferInput inferInput;

    @Before
    public void setUp() {
    }

    @Test
    public void testConstructorWithValidParameters() {
        String name = "input_tensor";
        long[] shape = {1, 3, 224, 224};
        TritonDataType datatype = TritonDataType.FP32;

        inferInput = new InferInput(name, shape, datatype);

        assertNotNull("InferInput should not be null", inferInput);
        assertEquals("Input name should match", name, inferInput.getName());
        assertArrayEquals("Shape should match", shape, inferInput.getShape());
        assertEquals("Datatype should match", datatype, inferInput.getDatatype());
    }

    @Test
    public void testConstructorWithIntDatatype() {
        String name = "int_input";
        long[] shape = {10, 20};
        TritonDataType datatype = TritonDataType.INT32;

        inferInput = new InferInput(name, shape, datatype);

        assertEquals("Name should be 'int_input'", "int_input", inferInput.getName());
        assertEquals("Datatype should be INT32", TritonDataType.INT32, inferInput.getDatatype());
    }

    @Test
    public void testConstructorWith1DShape() {
        String name = "flat_input";
        long[] shape = {100};
        TritonDataType datatype = TritonDataType.BOOL;

        inferInput = new InferInput(name, shape, datatype);

        assertEquals("Name should be 'flat_input'", "flat_input", inferInput.getName());
        assertArrayEquals("Shape should be [100]", shape, inferInput.getShape());
    }

    @Test
    public void testGetName() {
        String testName = "my_tensor";
        inferInput = new InferInput(testName, new long[]{1, 10}, TritonDataType.FP64);

        String retrievedName = inferInput.getName();

        assertEquals("Retrieved name should match constructor name", testName, retrievedName);
    }

    @Test
    public void testGetNameWithSpecialCharacters() {
        String testName = "tensor_123-abc";
        inferInput = new InferInput(testName, new long[]{5}, TritonDataType.INT64);

        assertEquals("Name with special characters should be preserved", testName, inferInput.getName());
    }

    @Test
    public void testGetDatatypeFP32() {
        inferInput = new InferInput("test", new long[]{1, 1}, TritonDataType.FP32);

        TritonDataType retrievedType = inferInput.getDatatype();

        assertEquals("Datatype should be FP32", TritonDataType.FP32, retrievedType);
    }

    @Test
    public void testGetDatatypeINT64() {
        inferInput = new InferInput("test", new long[]{2, 3}, TritonDataType.INT64);

        TritonDataType retrievedType = inferInput.getDatatype();

        assertEquals("Datatype should be INT64", TritonDataType.INT64, retrievedType);
    }

    @Test
    public void testGetDatatypeBYTES() {
        inferInput = new InferInput("strings", new long[]{10}, TritonDataType.BYTES);

        TritonDataType retrievedType = inferInput.getDatatype();

        assertEquals("Datatype should be BYTES", TritonDataType.BYTES, retrievedType);
    }

    @Test
    public void testGetDatatypeAllTypes() {
        TritonDataType[] allTypes = {
            TritonDataType.BOOL,
            TritonDataType.UINT8,
            TritonDataType.UINT16,
            TritonDataType.UINT32,
            TritonDataType.UINT64,
            TritonDataType.INT8,
            TritonDataType.INT16,
            TritonDataType.INT32,
            TritonDataType.INT64,
            TritonDataType.FP16,
            TritonDataType.FP32,
            TritonDataType.FP64,
            TritonDataType.BYTES
        };

        for (TritonDataType type : allTypes) {
            inferInput = new InferInput("test", new long[]{1}, type);
            assertEquals("Datatype should be " + type.toString(), type, inferInput.getDatatype());
        }
    }

    @Test
    public void testGetShape() {
        long[] expectedShape = {2, 3, 4, 5};
        inferInput = new InferInput("test", expectedShape, TritonDataType.FP32);

        long[] retrievedShape = inferInput.getShape();

        assertArrayEquals("Retrieved shape should match constructor shape", expectedShape, retrievedShape);
    }

    @Test
    public void testGetShape1D() {
        long[] expectedShape = {100};
        inferInput = new InferInput("test", expectedShape, TritonDataType.INT32);

        long[] retrievedShape = inferInput.getShape();

        assertArrayEquals("1D shape should be [100]", expectedShape, retrievedShape);
    }

    @Test
    public void testGetShapeLargeDimensions() {
        long[] expectedShape = {1, 256, 256, 3};
        inferInput = new InferInput("image_input", expectedShape, TritonDataType.UINT8);

        long[] retrievedShape = inferInput.getShape();

        assertArrayEquals("Shape should handle large dimensions", expectedShape, retrievedShape);
        assertEquals("Shape should have 4 dimensions", 4, retrievedShape.length);
    }

    @Test
    public void testGetShapeScalar() {
        long[] expectedShape = {1};
        inferInput = new InferInput("scalar", expectedShape, TritonDataType.FP32);

        long[] retrievedShape = inferInput.getShape();

        assertArrayEquals("Scalar shape should be [1]", expectedShape, retrievedShape);
        assertEquals("Scalar should have 1 element", 1, retrievedShape.length);
    }

    @Test
    public void testSetShape() {
        long[] initialShape = {10, 20};
        long[] newShape = {5, 40};
        inferInput = new InferInput("test", initialShape, TritonDataType.FP32);

        InferInput result = inferInput.setShape(newShape);

        assertArrayEquals("Shape should be updated to new shape", newShape, inferInput.getShape());
        assertSame("setShape() should return this for method chaining", inferInput, result);
    }

    @Test
    public void testSetShapeReturnsThis() {
        inferInput = new InferInput("test", new long[]{1, 1}, TritonDataType.FP32);

        InferInput result = inferInput.setShape(new long[]{2, 2});

        assertSame("setShape() should return the same object", inferInput, result);
    }

    @Test
    public void testSetShapeReplacement() {
        long[] initialShape = {10, 20, 30};
        long[] newShape = {5, 5};
        inferInput = new InferInput("test", initialShape, TritonDataType.INT64);

        inferInput.setShape(newShape);
        long[] actualShape = inferInput.getShape();

        assertArrayEquals("New shape should completely replace old shape", newShape, actualShape);
        assertEquals("New shape should have different number of dimensions", 2, actualShape.length);
    }

    @Test
    public void testSetShape1D() {
        inferInput = new InferInput("test", new long[]{1, 2, 3}, TritonDataType.FP32);
        long[] newShape = {60};

        inferInput.setShape(newShape);

        assertArrayEquals("Should be able to set 1D shape", newShape, inferInput.getShape());
    }

    @Test
    public void testSetShapeMultipleTimes() {
        long[] shape1 = {1, 10};
        long[] shape2 = {2, 20};
        long[] shape3 = {4, 5, 10};
        inferInput = new InferInput("test", shape1, TritonDataType.FP32);

        inferInput.setShape(shape2);
        assertArrayEquals("After first setShape", shape2, inferInput.getShape());

        inferInput.setShape(shape3);
        assertArrayEquals("After second setShape", shape3, inferInput.getShape());
    }

    @Test
    public void testSetShapeSingleElement() {
        inferInput = new InferInput("test", new long[]{10, 10}, TritonDataType.INT32);
        long[] newShape = {1};

        inferInput.setShape(newShape);

        assertArrayEquals("Should handle single element shape", newShape, inferInput.getShape());
    }

    @Test
    public void testSetShapeMethodChaining() {
        inferInput = new InferInput("test", new long[]{1, 1}, TritonDataType.FP32);
        long[] finalShape = {3, 4, 5};

        InferInput result = inferInput.setShape(finalShape);

        assertSame("Chain result should be same object", inferInput, result);
        assertArrayEquals("Final shape should be applied", finalShape, inferInput.getShape());
    }

    @Test
    public void testSetDataFloatSuccess() {
        inferInput = new InferInput("test", new long[]{2, 3}, TritonDataType.FP32);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this for method chaining", inferInput, result);
        assertNotNull("Raw content should not be null", inferInput.getRawContent());
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test(expected = Exception.class)
    public void testSetDataFloatWrongDatatype() {
        inferInput = new InferInput("test", new long[]{2, 3}, TritonDataType.INT32);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        inferInput.setData(data);
    }

    @Test(expected = Exception.class)
    public void testSetDataFloatWrongSize() {
        inferInput = new InferInput("test", new long[]{2, 3}, TritonDataType.FP32);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};

        inferInput.setData(data);
    }

    @Test
    public void testSetDataFloatImageRGB() {
        // Simulating a small RGB image of 3x3x3 (3x3 image with 3 channels)
        long[] imageShape = {3, 3, 3};
        inferInput = new InferInput("image_input", imageShape, TritonDataType.FP32);

        // Total elements: 3 * 3 * 3 = 27 floats
        float[] imageData = new float[27];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (float) i / 255.0f;
        }

        InferInput result = inferInput.setData(imageData);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content for image", inferInput.hasRawContent());
    }

    @Test
    public void testSetDataFloatLargeImage() {
        long[] imageShape = {1, 224, 224, 3}; // Batch of 1 image, 224x224 RGB
        inferInput = new InferInput("image_input", imageShape, TritonDataType.FP32);

        float[] imageData = new float[150528];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (float) (Math.random());
        }

        InferInput result = inferInput.setData(imageData);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content for large image", inferInput.hasRawContent());
    }

    @Test(expected = Exception.class)
    public void testSetDataFloatImageWrongSize() {
        long[] imageShape = {3, 3, 3};
        inferInput = new InferInput("image_input", imageShape, TritonDataType.FP32);

        float[] imageData = new float[20];

        inferInput.setData(imageData);
    }

    @Test
    public void testSetDataDoubleSuccess() {
        inferInput = new InferInput("test", new long[]{1, 4}, TritonDataType.FP64);
        double[] data = {1.5, 2.5, 3.5, 4.5};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this for method chaining", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test(expected = Exception.class)
    public void testSetDataDoubleWrongDatatype() {
        inferInput = new InferInput("test", new long[]{1, 4}, TritonDataType.FP32);
        double[] data = {1.5, 2.5, 3.5, 4.5};

        inferInput.setData(data);
    }

    @Test(expected = Exception.class)
    public void testSetDataDoubleWrongSize() {
        inferInput = new InferInput("test", new long[]{1, 4}, TritonDataType.FP64);
        double[] data = {1.5, 2.5, 3.5};

        inferInput.setData(data);
    }

    @Test
    public void testSetDataDoubleGrayscaleImage() {
        long[] imageShape = {256, 256};
        inferInput = new InferInput("grayscale_image", imageShape, TritonDataType.FP64);

        double[] imageData = new double[65536];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = Math.random();
        }

        InferInput result = inferInput.setData(imageData);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content for grayscale image", inferInput.hasRawContent());
    }

    @Test
    public void testSetDataIntSuccess() {
        inferInput = new InferInput("test", new long[]{2, 2}, TritonDataType.INT32);
        int[] data = {1, 2, 3, 4};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this for method chaining", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test
    public void testSetDataIntInt16Success() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.INT16);
        int[] data = {100, 200, 300};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test
    public void testSetDataIntInt8Success() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.INT8);
        int[] data = {1, 2, 3, 4, 5};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test(expected = Exception.class)
    public void testSetDataIntWrongDatatype() {
        inferInput = new InferInput("test", new long[]{2, 2}, TritonDataType.FP32);
        int[] data = {1, 2, 3, 4};

        inferInput.setData(data);
    }

    @Test(expected = Exception.class)
    public void testSetDataIntWrongSize() {
        inferInput = new InferInput("test", new long[]{2, 2}, TritonDataType.INT32);
        int[] data = {1, 2, 3};

        inferInput.setData(data);
    }

    @Test
    public void testSetDataIntSegmentationMask() {
        long[] maskShape = {32, 32};
        inferInput = new InferInput("segmentation_mask", maskShape, TritonDataType.INT32);

        int[] maskData = new int[1024];
        for (int i = 0; i < maskData.length; i++) {
            maskData[i] = i % 10;
        }

        InferInput result = inferInput.setData(maskData);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content for mask", inferInput.hasRawContent());
    }

    @Test
    public void testSetDataLongSuccess() {
        inferInput = new InferInput("test", new long[]{1, 3}, TritonDataType.INT64);
        long[] data = {1000000000L, 2000000000L, 3000000000L};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this for method chaining", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test(expected = Exception.class)
    public void testSetDataLongWrongDatatype() {
        inferInput = new InferInput("test", new long[]{1, 3}, TritonDataType.INT32);
        long[] data = {1000000000L, 2000000000L, 3000000000L};

        inferInput.setData(data);
    }

    @Test(expected = Exception.class)
    public void testSetDataLongWrongSize() {
        inferInput = new InferInput("test", new long[]{1, 3}, TritonDataType.INT64);
        long[] data = {1000000000L, 2000000000L};

        inferInput.setData(data);
    }

    @Test
    public void testSetDataBooleanSuccess() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.BOOL);
        boolean[] data = {true, false, true, false};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this for method chaining", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test(expected = Exception.class)
    public void testSetDataBooleanWrongDatatype() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.INT32);
        boolean[] data = {true, false, true, false};

        inferInput.setData(data);
    }

    @Test(expected = Exception.class)
    public void testSetDataBooleanWrongSize() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.BOOL);
        boolean[] data = {true, false, true}; // Only 3 elements instead of 4

        inferInput.setData(data);
    }

    @Test
    public void testSetDataByteSuccess() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.INT32);
        byte[] data = {1, 2, 3, 4, 5};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this for method chaining", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
        assertArrayEquals("Raw content should match input", data, inferInput.getRawContent());
    }

    @Test
    public void testSetDataByteEmpty() {
        inferInput = new InferInput("test", new long[]{0}, TritonDataType.INT32);
        byte[] data = new byte[0];

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this", inferInput, result);
        assertFalse("Empty byte array should not have raw content", inferInput.hasRawContent());
    }

    @Test
    public void testSetDataByteRawImageBytes() {
        inferInput = new InferInput("encoded_image", new long[]{12345}, TritonDataType.UINT8);
        byte[] imageBytes = new byte[12345];
        for (int i = 0; i < imageBytes.length; i++) {
            imageBytes[i] = (byte) (i % 256);
        }

        InferInput result = inferInput.setData(imageBytes);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content for encoded image", inferInput.hasRawContent());
        assertEquals("Byte array size should match", imageBytes.length, inferInput.getRawContent().length);
    }

    @Test
    public void testSetDataStringSuccess() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.BYTES);
        String[] data = {"hello", "world", "test"};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this for method chaining", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test(expected = Exception.class)
    public void testSetDataStringWrongDatatype() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.FP32);
        String[] data = {"hello", "world", "test"};

        inferInput.setData(data);
    }

    @Test(expected = Exception.class)
    public void testSetDataStringWrongSize() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.BYTES);
        String[] data = {"hello", "world"}; // Only 2 elements instead of 3

        inferInput.setData(data);
    }

    @Test
    public void testSetDataStringSpecialCharacters() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.BYTES);
        String[] data = {"Héllo", "café"};

        InferInput result = inferInput.setData(data);

        assertSame("setData should return this", inferInput, result);
        assertTrue("Should have raw content", inferInput.hasRawContent());
    }

    @Test
    public void testSetDataClearsSharedMemoryParams() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.FP32);
        float[] data = {1.0f, 2.0f};

        inferInput.setData(data);

        assertTrue("Should have raw content after setData", inferInput.hasRawContent());
    }

    @Test
    public void testGetTensorName() {
        String testName = "input_tensor";
        inferInput = new InferInput(testName, new long[]{10}, TritonDataType.FP32);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null", tensor);
        assertEquals("Tensor name should match", testName, inferInput.getName());
    }

    @Test
    public void testGetTensorDatatype() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.INT32);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null", tensor);
        assertEquals("Tensor datatype should match", TritonDataType.INT32, inferInput.getDatatype());
    }

    @Test
    public void testGetTensorShape() {
        long[] expectedShape = {2, 3, 4};
        inferInput = new InferInput("test", expectedShape, TritonDataType.FP64);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null", tensor);
        assertArrayEquals("Tensor shape should match", expectedShape, inferInput.getShape());
    }

    @Test
    public void testGetTensorWithBoolDatatype() {
        inferInput = new InferInput("bool_input", new long[]{8}, TritonDataType.BOOL);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null for BOOL", tensor);
        assertEquals("Tensor datatype should be BOOL", TritonDataType.BOOL, inferInput.getDatatype());
    }

    @Test
    public void testGetTensorWithBytesDatatype() {
        inferInput = new InferInput("string_input", new long[]{5}, TritonDataType.BYTES);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null for BYTES", tensor);
        assertEquals("Tensor datatype should be BYTES", TritonDataType.BYTES, inferInput.getDatatype());
    }

    @Test
    public void testGetTensorWithImageShape() {
        long[] imageShape = {1, 224, 224, 3};
        inferInput = new InferInput("image_input", imageShape, TritonDataType.UINT8);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null", tensor);
        long[] retrievedShape = inferInput.getShape();
        assertArrayEquals("Tensor should have image shape", imageShape, retrievedShape);
    }

    @Test
    public void testGetTensorAfterSetData() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.FP32);
        float[] data = {1.0f, 2.0f, 3.0f};
        inferInput.setData(data);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null after setData", tensor);
        assertTrue("Should have raw content after setData", inferInput.hasRawContent());
    }

    @Test
    public void testGetTensorScalar() {
        inferInput = new InferInput("scalar_input", new long[]{1}, TritonDataType.INT64);

        Object tensor = inferInput.getTensor();

        assertNotNull("getTensor should not return null for scalar", tensor);
        assertEquals("Scalar should have 1 element", 1, inferInput.getShape().length);
    }

    @Test
    public void testGetRawContentNoData() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.FP32);

        byte[] rawContent = inferInput.getRawContent();

        assertNull("getRawContent should return null when no data is set", rawContent);
    }

    @Test
    public void testGetRawContentAfterSetFloatData() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.FP32);
        float[] data = {1.5f, 2.5f};
        inferInput.setData(data);

        byte[] rawContent = inferInput.getRawContent();

        assertNotNull("getRawContent should not be null after setData", rawContent);
        assertTrue("Raw content should have data", rawContent.length > 0);
        assertEquals("Raw content should be 8 bytes for 2 floats", 8, rawContent.length);
    }

    @Test
    public void testGetRawContentAfterSetIntData() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.INT32);
        int[] data = {100, 200, 300};
        inferInput.setData(data);

        byte[] rawContent = inferInput.getRawContent();

        assertNotNull("getRawContent should not be null after setData", rawContent);
        assertEquals("Raw content should be 12 bytes for 3 ints", 12, rawContent.length);
    }

    @Test
    public void testGetRawContentAfterSetByteData() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.UINT8);
        byte[] data = {1, 2, 3, 4, 5};
        inferInput.setData(data);

        byte[] rawContent = inferInput.getRawContent();

        assertNotNull("getRawContent should not be null after setData", rawContent);
        assertArrayEquals("Raw content should match input byte array", data, rawContent);
    }

    @Test
    public void testGetRawContentAfterSetStringData() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.BYTES);
        String[] data = {"hello", "world"};
        inferInput.setData(data);

        byte[] rawContent = inferInput.getRawContent();

        assertNotNull("getRawContent should not be null after setData", rawContent);
        assertTrue("Raw content should have data for strings", rawContent.length > 0);
    }

    @Test
    public void testGetRawContentLargeImageData() {
        long[] imageShape = {1, 256, 256, 3};
        inferInput = new InferInput("image", imageShape, TritonDataType.UINT8);
        byte[] imageData = new byte[196608]; // 1 * 256 * 256 * 3
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (byte) (i % 256);
        }
        inferInput.setData(imageData);

        byte[] rawContent = inferInput.getRawContent();

        assertNotNull("getRawContent should not be null for image data", rawContent);
        assertEquals("Raw content should match image size", imageData.length, rawContent.length);
    }

    @Test
    public void testGetRawContentAfterSetDoubleData() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.FP64);
        double[] data = {1.1, 2.2, 3.3, 4.4};
        inferInput.setData(data);

        byte[] rawContent = inferInput.getRawContent();

        assertNotNull("getRawContent should not be null after setData", rawContent);
        assertEquals("Raw content should be 32 bytes for 4 doubles", 32, rawContent.length);
    }

    @Test
    public void testHasRawContentInitiallyFalse() {
        inferInput = new InferInput("test", new long[]{10}, TritonDataType.FP32);

        boolean hasContent = inferInput.hasRawContent();

        assertFalse("hasRawContent should be false initially", hasContent);
    }

    @Test
    public void testHasRawContentAfterSetFloatData() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.FP32);
        float[] data = {1.0f, 2.0f, 3.0f};

        inferInput.setData(data);

        assertTrue("hasRawContent should be true after setData", inferInput.hasRawContent());
    }

    @Test
    public void testHasRawContentAfterSetIntData() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.INT32);
        int[] data = {1, 2, 3, 4, 5};

        inferInput.setData(data);

        assertTrue("hasRawContent should be true after setData", inferInput.hasRawContent());
    }

    @Test
    public void testHasRawContentEmptyByteArray() {
        inferInput = new InferInput("test", new long[]{0}, TritonDataType.INT32);
        byte[] emptyData = new byte[0];

        inferInput.setData(emptyData);

        assertFalse("hasRawContent should be false for empty byte array", inferInput.hasRawContent());
    }

    @Test
    public void testHasRawContentNonEmptyByteArray() {
        inferInput = new InferInput("test", new long[]{1}, TritonDataType.UINT8);
        byte[] data = {42};

        inferInput.setData(data);

        assertTrue("hasRawContent should be true for non-empty byte array", inferInput.hasRawContent());
    }

    @Test
    public void testHasRawContentAfterSetBooleanData() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.BOOL);
        boolean[] data = {true, false, true, false};

        inferInput.setData(data);

        assertTrue("hasRawContent should be true after setData", inferInput.hasRawContent());
    }

    @Test
    public void testHasRawContentAfterSetStringData() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.BYTES);
        String[] data = {"hello", "world", "test"};

        inferInput.setData(data);

        assertTrue("hasRawContent should be true after setData with strings", inferInput.hasRawContent());
    }

    @Test
    public void testHasRawContentWithImageData() {
        long[] imageShape = {224, 224, 3};
        inferInput = new InferInput("image", imageShape, TritonDataType.FP32);
        float[] imageData = new float[150528];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (float) i / 150528.0f;
        }

        inferInput.setData(imageData);

        assertTrue("hasRawContent should be true for image data", inferInput.hasRawContent());
    }

    @Test
    public void testHasRawContentAfterSetLongData() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.INT64);
        long[] data = {1000000000L, 2000000000L};

        inferInput.setData(data);

        assertTrue("hasRawContent should be true after setData with long array", inferInput.hasRawContent());
    }

    @Test
    public void testGetDataAsFloatArraySuccess() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.FP32);
        float[] originalData = {1.5f, 2.5f, 3.5f};
        inferInput.setData(originalData);

        float[] retrievedData = inferInput.getDataAsFloatArray();

        assertNotNull("Retrieved data should not be null", retrievedData);
        assertArrayEquals("Retrieved data should match original", originalData, retrievedData, 0.001f);
    }

    @Test(expected = Exception.class)
    public void testGetDataAsFloatArrayNoData() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.FP32);

        inferInput.getDataAsFloatArray();
    }

    @Test
    public void testGetDataAsFloatArrayImageData() {
        long[] imageShape = {3, 3, 3};
        inferInput = new InferInput("image", imageShape, TritonDataType.FP32);
        float[] imageData = new float[27];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (float) i / 27.0f;
        }
        inferInput.setData(imageData);

        float[] retrievedData = inferInput.getDataAsFloatArray();

        assertEquals("Retrieved array length should match", imageData.length, retrievedData.length);
        assertArrayEquals("Image data should match", imageData, retrievedData, 0.001f);
    }

    @Test
    public void testGetDataAsFloatArraySingleElement() {
        inferInput = new InferInput("test", new long[]{1}, TritonDataType.FP32);
        float[] data = {42.5f};
        inferInput.setData(data);

        float[] retrievedData = inferInput.getDataAsFloatArray();

        assertEquals("Should retrieve single element", 1, retrievedData.length);
        assertEquals("Value should match", 42.5f, retrievedData[0], 0.001f);
    }

    @Test
    public void testGetDataAsDoubleArraySuccess() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.FP64);
        double[] originalData = {1.1111, 2.2222, 3.3333, 4.4444};
        inferInput.setData(originalData);

        double[] retrievedData = inferInput.getDataAsDoubleArray();

        assertNotNull("Retrieved data should not be null", retrievedData);
        assertArrayEquals("Retrieved data should match original", originalData, retrievedData, 0.0001);
    }

    @Test(expected = Exception.class)
    public void testGetDataAsDoubleArrayNoData() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.FP64);

        inferInput.getDataAsDoubleArray();
    }

    @Test
    public void testGetDataAsDoubleArrayGrayscaleImage() {
        long[] imageShape = {128, 128};
        inferInput = new InferInput("grayscale", imageShape, TritonDataType.FP64);
        double[] imageData = new double[16384];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = Math.random();
        }
        inferInput.setData(imageData);

        double[] retrievedData = inferInput.getDataAsDoubleArray();

        assertEquals("Retrieved array length should match", imageData.length, retrievedData.length);
        assertArrayEquals("Grayscale data should match", imageData, retrievedData, 0.0001);
    }

    @Test
    public void testGetDataAsIntArraySuccess() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.INT32);
        int[] originalData = {100, 200, 300, 400, 500};
        inferInput.setData(originalData);

        int[] retrievedData = inferInput.getDataAsIntArray();

        assertNotNull("Retrieved data should not be null", retrievedData);
        assertArrayEquals("Retrieved data should match original", originalData, retrievedData);
    }

    @Test(expected = Exception.class)
    public void testGetDataAsIntArrayNoData() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.INT32);

        inferInput.getDataAsIntArray();
    }

    @Test
    public void testGetDataAsIntArraySegmentationMask() {
        long[] maskShape = {64, 64};
        inferInput = new InferInput("mask", maskShape, TritonDataType.INT32);
        int[] maskData = new int[4096];
        for (int i = 0; i < maskData.length; i++) {
            maskData[i] = i % 10;
        }
        inferInput.setData(maskData);

        int[] retrievedData = inferInput.getDataAsIntArray();

        assertEquals("Retrieved array length should match", maskData.length, retrievedData.length);
        assertArrayEquals("Mask data should match", maskData, retrievedData);
    }

    @Test
    public void testGetDataAsIntArrayNegativeValues() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.INT32);
        int[] data = {-100, 0, 100};
        inferInput.setData(data);

        int[] retrievedData = inferInput.getDataAsIntArray();

        assertArrayEquals("Should handle negative values", data, retrievedData);
    }

    @Test
    public void testGetDataAsLongArraySuccess() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.INT64);
        long[] originalData = {1000000000L, 9999999999L};
        inferInput.setData(originalData);

        long[] retrievedData = inferInput.getDataAsLongArray();

        assertNotNull("Retrieved data should not be null", retrievedData);
        assertArrayEquals("Retrieved data should match original", originalData, retrievedData);
    }

    @Test(expected = Exception.class)
    public void testGetDataAsLongArrayNoData() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.INT64);

        inferInput.getDataAsLongArray();
    }

    @Test
    public void testGetDataAsLongArrayLargeValues() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.INT64);
        long[] data = {Long.MAX_VALUE, 0, Long.MIN_VALUE};
        inferInput.setData(data);

        long[] retrievedData = inferInput.getDataAsLongArray();

        assertArrayEquals("Should handle extreme values", data, retrievedData);
    }

    @Test
    public void testGetDataAsBooleanArraySuccess() {
        inferInput = new InferInput("test", new long[]{6}, TritonDataType.BOOL);
        boolean[] originalData = {true, false, true, true, false, false};
        inferInput.setData(originalData);

        boolean[] retrievedData = inferInput.getDataAsBooleanArray();

        assertNotNull("Retrieved data should not be null", retrievedData);
        assertEquals("Array length should match", originalData.length, retrievedData.length);
        for (int i = 0; i < originalData.length; i++) {
            assertEquals("Element " + i + " should match", originalData[i], retrievedData[i]);
        }
    }

    @Test(expected = Exception.class)
    public void testGetDataAsBooleanArrayNoData() {
        inferInput = new InferInput("test", new long[]{6}, TritonDataType.BOOL);

        inferInput.getDataAsBooleanArray();
    }

    @Test
    public void testGetDataAsBooleanArrayAllTrue() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.BOOL);
        boolean[] data = {true, true, true, true};
        inferInput.setData(data);

        boolean[] retrievedData = inferInput.getDataAsBooleanArray();

        assertEquals("All values should be true", data.length, retrievedData.length);
        for (int i = 0; i < data.length; i++) {
            assertEquals("Element " + i + " should be true", data[i], retrievedData[i]);
        }
    }

    @Test
    public void testGetDataAsBooleanArrayAllFalse() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.BOOL);
        boolean[] data = {false, false, false, false};
        inferInput.setData(data);

        boolean[] retrievedData = inferInput.getDataAsBooleanArray();

        assertEquals("Array length should match", data.length, retrievedData.length);
        for (int i = 0; i < data.length; i++) {
            assertEquals("Element " + i + " should be false", data[i], retrievedData[i]);
        }
    }

    @Test
    public void testGetDataAsStringArraySuccess() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.BYTES);
        String[] originalData = {"hello", "world", "test"};
        inferInput.setData(originalData);

        String[] retrievedData = inferInput.getDataAsStringArray();

        assertNotNull("Retrieved data should not be null", retrievedData);
        assertArrayEquals("Retrieved data should match original", originalData, retrievedData);
    }

    @Test(expected = Exception.class)
    public void testGetDataAsStringArrayNoData() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.BYTES);

        inferInput.getDataAsStringArray();
    }

    @Test
    public void testGetDataAsStringArraySpecialCharacters() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.BYTES);
        String[] originalData = {"Héllo", "café", "naïve", "日本語"};
        inferInput.setData(originalData);

        String[] retrievedData = inferInput.getDataAsStringArray();

        assertArrayEquals("Should handle special characters", originalData, retrievedData);
    }

    @Test
    public void testGetDataAsStringArrayEmptyStrings() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.BYTES);
        String[] originalData = {"", "hello", ""};
        inferInput.setData(originalData);

        String[] retrievedData = inferInput.getDataAsStringArray();

        assertArrayEquals("Should handle empty strings", originalData, retrievedData);
    }

    @Test
    public void testGetDataAsStringArraySingleString() {
        inferInput = new InferInput("test", new long[]{1}, TritonDataType.BYTES);
        String[] originalData = {"single_string"};
        inferInput.setData(originalData);

        String[] retrievedData = inferInput.getDataAsStringArray();

        assertEquals("Should retrieve single string", 1, retrievedData.length);
        assertEquals("String should match", "single_string", retrievedData[0]);
    }

    @Test
    public void testGetDataAsStringArrayLongStrings() {
        inferInput = new InferInput("test", new long[]{2}, TritonDataType.BYTES);
        String longString1 = "A".repeat(1000);
        String longString2 = "B".repeat(1000);
        String[] originalData = {longString1, longString2};
        inferInput.setData(originalData);

        String[] retrievedData = inferInput.getDataAsStringArray();

        assertArrayEquals("Should handle long strings", originalData, retrievedData);
    }

    @Test
    public void testGetDataAsStringArrayManyStrings() {
        int count = 100;
        inferInput = new InferInput("test", new long[]{count}, TritonDataType.BYTES);
        String[] originalData = new String[count];
        for (int i = 0; i < count; i++) {
            originalData[i] = "string_" + i;
        }
        inferInput.setData(originalData);

        String[] retrievedData = inferInput.getDataAsStringArray();

        assertEquals("Should retrieve all strings", count, retrievedData.length);
        assertArrayEquals("All strings should match", originalData, retrievedData);
    }

    @Test
    public void testRoundTripFloat() {
        inferInput = new InferInput("test", new long[]{5}, TritonDataType.FP32);
        float[] originalData = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
        inferInput.setData(originalData);

        float[] retrievedData = inferInput.getDataAsFloatArray();

        assertArrayEquals("Round-trip should preserve float data", originalData, retrievedData, 0.001f);
    }

    @Test
    public void testRoundTripInt() {
        inferInput = new InferInput("test", new long[]{4}, TritonDataType.INT32);
        int[] originalData = {-1000, 0, 1000, 2000};
        inferInput.setData(originalData);

        int[] retrievedData = inferInput.getDataAsIntArray();

        assertArrayEquals("Round-trip should preserve int data", originalData, retrievedData);
    }

    @Test
    public void testRoundTripString() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.BYTES);
        String[] originalData = {"first", "second", "third"};
        inferInput.setData(originalData);

        String[] retrievedData = inferInput.getDataAsStringArray();

        assertArrayEquals("Round-trip should preserve string data", originalData, retrievedData);
    }

    @Test
    public void testRoundTripBoolean() {
        inferInput = new InferInput("test", new long[]{8}, TritonDataType.BOOL);
        boolean[] originalData = {true, false, true, false, true, false, true, false};
        inferInput.setData(originalData);

        boolean[] retrievedData = inferInput.getDataAsBooleanArray();

        assertEquals("Array length should match", originalData.length, retrievedData.length);
        for (int i = 0; i < originalData.length; i++) {
            assertEquals("Element " + i + " should match", originalData[i], retrievedData[i]);
        }
    }

    @Test
    public void testRoundTripDouble() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.FP64);
        double[] originalData = {1.123456789, 2.987654321, 3.555555555};
        inferInput.setData(originalData);

        double[] retrievedData = inferInput.getDataAsDoubleArray();

        assertArrayEquals("Round-trip should preserve double data", originalData, retrievedData, 0.000000001);
    }

    @Test
    public void testRoundTripLong() {
        inferInput = new InferInput("test", new long[]{3}, TritonDataType.INT64);
        long[] originalData = {-999999999L, 0L, 999999999L};
        inferInput.setData(originalData);

        long[] retrievedData = inferInput.getDataAsLongArray();

        assertArrayEquals("Round-trip should preserve long data", originalData, retrievedData);
    }
}
