package com.technodrome.diffusion.network;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-layer convolutional network with multi-scale convolutions.
 *
 * Stacks multiple multi-scale convolution layers with leaky ReLU activations.
 * Optionally includes layer normalization or batch statistics.
 */
public class MultiLayerConvolution extends AbstractBlock {

    private static final byte VERSION = 1;

    private final int inputChannels;
    private final int hiddenChannels;
    private final int outputChannels;
    private final int numLayers;
    private final int numScales;
    private final int kernelSize;
    private final float leakySlope;

    private final List<MultiScaleConvolution> convLayers;
    private final List<Parameter> layerBiases;

    /**
     * Create a multi-layer convolutional network.
     *
     * @param inputChannels Number of input channels
     * @param hiddenChannels Number of channels in hidden layers
     * @param outputChannels Number of output channels
     * @param numLayers Number of conv layers
     * @param numScales Number of scales for multi-scale convolution
     * @param kernelSize Convolution kernel size
     */
    public MultiLayerConvolution(int inputChannels, int hiddenChannels, int outputChannels,
                                  int numLayers, int numScales, int kernelSize) {
        this(inputChannels, hiddenChannels, outputChannels, numLayers, numScales, kernelSize, 0.05f);
    }

    public MultiLayerConvolution(int inputChannels, int hiddenChannels, int outputChannels,
                                  int numLayers, int numScales, int kernelSize, float leakySlope) {
        super(VERSION);
        this.inputChannels = inputChannels;
        this.hiddenChannels = hiddenChannels;
        this.outputChannels = outputChannels;
        this.numLayers = numLayers;
        this.numScales = numScales;
        this.kernelSize = kernelSize;
        this.leakySlope = leakySlope;

        this.convLayers = new ArrayList<>();
        this.layerBiases = new ArrayList<>();

        // Create layers
        int prevChannels = inputChannels;
        for (int i = 0; i < numLayers; i++) {
            int currOutputChannels;
            if (i == numLayers - 1) {
                // Last layer outputs the desired output channels
                currOutputChannels = outputChannels / numScales;  // Per-scale output
            } else {
                currOutputChannels = hiddenChannels / numScales;  // Per-scale hidden
            }

            MultiScaleConvolution layer = new MultiScaleConvolution(
                    prevChannels, currOutputChannels, numScales, kernelSize);
            addChildBlock("conv" + i, layer);
            convLayers.add(layer);

            // Bias for residual or normalization
            Parameter bias = addParameter(
                    Parameter.builder()
                            .setName("layer_bias" + i)
                            .setType(Parameter.Type.BIAS)
                            .build());
            layerBiases.add(bias);

            prevChannels = currOutputChannels * numScales;
        }
    }

    @Override
    public void prepare(Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        long height = inputShape.get(2);
        long width = inputShape.get(3);

        // Prepare each conv layer
        int prevChannels = inputChannels;
        for (int i = 0; i < numLayers; i++) {
            int currOutputChannels;
            if (i == numLayers - 1) {
                currOutputChannels = outputChannels / numScales;
            } else {
                currOutputChannels = hiddenChannels / numScales;
            }

            Shape[] layerInputShapes = new Shape[]{new Shape(1, prevChannels, height, width)};
            convLayers.get(i).prepare(layerInputShapes);

            layerBiases.get(i).setShape(new Shape(currOutputChannels * numScales));

            prevChannels = currOutputChannels * numScales;
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

        for (int i = 0; i < numLayers; i++) {
            // Apply multi-scale convolution
            NDList convOutput = convLayers.get(i).forward(parameterStore, new NDList(x), training);
            x = convOutput.singletonOrThrow();

            // Add bias (broadcast over spatial dimensions)
            NDArray bias = parameterStore.getValue(layerBiases.get(i), x.getDevice(), training);
            x = x.add(bias.reshape(1, -1, 1, 1));

            // Apply leaky ReLU (except for last layer if needed)
            if (i < numLayers - 1) {
                x = LeakyRelu.apply(x, leakySlope);
            }
        }

        return new NDList(x);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        return new Shape[]{
                new Shape(inputShape.get(0), outputChannels, inputShape.get(2), inputShape.get(3))
        };
    }

    public int getInputChannels() {
        return inputChannels;
    }

    public int getHiddenChannels() {
        return hiddenChannels;
    }

    public int getOutputChannels() {
        return outputChannels;
    }

    public int getNumLayers() {
        return numLayers;
    }
}
