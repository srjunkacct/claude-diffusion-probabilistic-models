package com.technodrome.diffusion.network;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-scale convolution layer using DJL's built-in Conv2d.
 *
 * Processes input at multiple spatial scales:
 * 1. Downsamples input to each scale using mean pooling
 * 2. Applies convolution at each scale
 * 3. Upsamples back to original resolution
 * 4. Concatenates all scale outputs
 *
 * This allows the network to capture both local and global features.
 */
public class MultiScaleConvolution extends AbstractBlock {

    private static final byte VERSION = 1;

    private final int inputChannels;
    private final int outputChannels;
    private final int numScales;
    private final int kernelSize;
    private final List<Conv2d> convLayers;

    /**
     * Create a multi-scale convolution layer.
     *
     * @param inputChannels Number of input channels
     * @param outputChannels Number of output channels per scale
     * @param numScales Number of scales (1 = original only, 2 = original + 2x downsampled, etc.)
     * @param kernelSize Convolution kernel size
     */
    public MultiScaleConvolution(int inputChannels, int outputChannels, int numScales, int kernelSize) {
        super(VERSION);
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.numScales = numScales;
        this.kernelSize = kernelSize;

        this.convLayers = new ArrayList<>();

        // Create Conv2d block for each scale with Xavier initialization
        XavierInitializer xavier = new XavierInitializer(XavierInitializer.RandomType.GAUSSIAN, XavierInitializer.FactorType.IN, 2.0f);
        int padding = kernelSize / 2;  // Same padding to preserve spatial dimensions
        for (int s = 0; s < numScales; s++) {
            Conv2d conv = Conv2d.builder()
                    .setFilters(outputChannels)
                    .setKernelShape(new Shape(kernelSize, kernelSize))
                    .optPadding(new Shape(padding, padding))
                    .optBias(true)
                    .build();
            conv.setInitializer(xavier, Parameter.Type.WEIGHT);
            convLayers.add(conv);
            addChildBlock("conv_scale" + s, conv);
        }
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape inputShape = inputShapes[0];

        for (int s = 0; s < numScales; s++) {
            // Calculate input shape at this scale (downsampled by 2^s)
            long height = inputShape.get(2) >> s;  // Divide by 2^s
            long width = inputShape.get(3) >> s;
            Shape scaledShape = new Shape(inputShape.get(0), inputChannels,
                    Math.max(1, height), Math.max(1, width));
            convLayers.get(s).initialize(manager, dataType, scaledShape);
        }
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {

        NDArray x = inputs.singletonOrThrow();
        NDManager manager = x.getManager();
        Shape originalShape = x.getShape();
        long batch = originalShape.get(0);
        long height = originalShape.get(2);
        long width = originalShape.get(3);

        List<NDArray> scaleOutputs = new ArrayList<>();

        for (int s = 0; s < numScales; s++) {
            NDArray scaledInput = x;

            // Downsample by 2^s using mean pooling
            for (int i = 0; i < s; i++) {
                scaledInput = meanPool2x2(scaledInput);
            }

            // Apply convolution at this scale using built-in Conv2d
            NDList convOutput = convLayers.get(s).forward(parameterStore, new NDList(scaledInput), training);
            NDArray output = convOutput.singletonOrThrow();

            // Upsample back to original resolution
            for (int i = 0; i < s; i++) {
                output = upsample2x(output);
            }

            // Ensure output matches original spatial dimensions
            if (output.getShape().get(2) != height || output.getShape().get(3) != width) {
                output = adjustSpatialSize(output, height, width, manager);
            }

            scaleOutputs.add(output);
        }

        // Concatenate all scales along channel dimension
        NDArray result = scaleOutputs.get(0);
        for (int s = 1; s < numScales; s++) {
            result = result.concat(scaleOutputs.get(s), 1);
        }

        return new NDList(result);
    }

    private NDArray meanPool2x2(NDArray x) {
        Shape shape = x.getShape();
        long batch = shape.get(0);
        long channels = shape.get(1);
        long height = shape.get(2);
        long width = shape.get(3);

        // Ensure even dimensions
        long newHeight = (height / 2) * 2;
        long newWidth = (width / 2) * 2;

        if (newHeight != height || newWidth != width) {
            x = x.get(":, :, :" + newHeight + ", :" + newWidth);
        }

        // Reshape and mean pool
        NDArray reshaped = x.reshape(batch, channels, newHeight / 2, 2, newWidth / 2, 2);
        return reshaped.mean(new int[]{5}).mean(new int[]{3});
    }

    private NDArray upsample2x(NDArray x) {
        Shape shape = x.getShape();
        long batch = shape.get(0);
        long channels = shape.get(1);
        long height = shape.get(2);
        long width = shape.get(3);

        // Reshape to (batch, channels, height, 1, width, 1) then tile
        NDArray expanded = x.reshape(batch, channels, height, 1, width, 1);
        NDArray tiled = expanded.tile(new long[]{1, 1, 1, 2, 1, 2});
        return tiled.reshape(batch, channels, height * 2, width * 2);
    }

    private NDArray adjustSpatialSize(NDArray x, long targetHeight, long targetWidth, NDManager manager) {
        Shape shape = x.getShape();
        long currentHeight = shape.get(2);
        long currentWidth = shape.get(3);

        if (currentHeight > targetHeight) {
            long startH = (currentHeight - targetHeight) / 2;
            x = x.get(":, :, " + startH + ":" + (startH + targetHeight) + ", :");
        }
        if (currentWidth > targetWidth) {
            long startW = (currentWidth - targetWidth) / 2;
            x = x.get(":, :, :, " + startW + ":" + (startW + targetWidth));
        }

        return x;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        return new Shape[]{
                new Shape(inputShape.get(0), (long) outputChannels * numScales,
                        inputShape.get(2), inputShape.get(3))
        };
    }

    public int getInputChannels() {
        return inputChannels;
    }

    public int getOutputChannels() {
        return outputChannels;
    }

    public int getNumScales() {
        return numScales;
    }

    public int getTotalOutputChannels() {
        return outputChannels * numScales;
    }
}
