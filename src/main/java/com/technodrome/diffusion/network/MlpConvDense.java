package com.technodrome.diffusion.network;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;

/**
 * Main MLP architecture for the diffusion model using DJL's built-in Conv2d.
 *
 * Architecture consists of two parts:
 * 1. Lower MLP: Convolutional layers for feature extraction
 * 2. Upper MLP: Per-pixel dense layers (1x1 convolutions) for temporal coefficient prediction
 *
 * Outputs temporal coefficients for both mu (mean) and sigma (std) of the reverse process.
 */
public class MlpConvDense extends AbstractBlock {

    private static final byte VERSION = 1;

    // Architecture hyperparameters
    private final int inputChannels;
    private final int numTemporalBasis;
    private final int lowerHiddenChannels;
    private final int upperHiddenChannels;
    private final int numLowerLayers;
    private final int numUpperLayers;
    private final int numScales;
    private final int kernelSize;
    private final float leakySlope;

    // Network components
    private final MultiLayerConvolution lowerMlp;
    private final List<Conv2d> upperConvLayers;
    private final Conv2d muConv;
    private final Conv2d sigmaConv;

    /**
     * Create the main MLP network.
     *
     * @param inputChannels Number of input image channels (e.g., 1 for MNIST)
     * @param numTemporalBasis Number of temporal basis functions
     * @param lowerHiddenChannels Hidden channels in lower conv layers
     * @param upperHiddenChannels Hidden channels in upper dense layers
     * @param numLowerLayers Number of lower conv layers
     * @param numUpperLayers Number of upper dense layers
     * @param numScales Number of scales for multi-scale convolution
     * @param kernelSize Convolution kernel size
     */
    public MlpConvDense(int inputChannels, int numTemporalBasis,
                        int lowerHiddenChannels, int upperHiddenChannels,
                        int numLowerLayers, int numUpperLayers,
                        int numScales, int kernelSize) {
        super(VERSION);
        this.inputChannels = inputChannels;
        this.numTemporalBasis = numTemporalBasis;
        this.lowerHiddenChannels = lowerHiddenChannels;
        this.upperHiddenChannels = upperHiddenChannels;
        this.numLowerLayers = numLowerLayers;
        this.numUpperLayers = numUpperLayers;
        this.numScales = numScales;
        this.kernelSize = kernelSize;
        this.leakySlope = 0.05f;

        // Create lower MLP (convolutional)
        this.lowerMlp = new MultiLayerConvolution(
                inputChannels,
                lowerHiddenChannels,
                lowerHiddenChannels,
                numLowerLayers,
                numScales,
                kernelSize,
                leakySlope);
        addChildBlock("lower_mlp", lowerMlp);

        // Create upper MLP using 1x1 Conv2d blocks with Xavier initialization
        XavierInitializer xavier = new XavierInitializer(XavierInitializer.RandomType.GAUSSIAN, XavierInitializer.FactorType.IN, 2.0f);

        this.upperConvLayers = new ArrayList<>();
        for (int i = 0; i < numUpperLayers; i++) {
            Conv2d conv = Conv2d.builder()
                    .setFilters(upperHiddenChannels)
                    .setKernelShape(new Shape(1, 1))
                    .optBias(true)
                    .build();
            conv.setInitializer(xavier, ai.djl.nn.Parameter.Type.WEIGHT);
            upperConvLayers.add(conv);
            addChildBlock("upper_conv" + i, conv);
        }

        // Output layers for mu and sigma coefficients
        int outputChannels = inputChannels * numTemporalBasis;

        this.muConv = Conv2d.builder()
                .setFilters(outputChannels)
                .setKernelShape(new Shape(1, 1))
                .optBias(true)
                .build();
        muConv.setInitializer(xavier, ai.djl.nn.Parameter.Type.WEIGHT);
        addChildBlock("mu_conv", muConv);

        this.sigmaConv = Conv2d.builder()
                .setFilters(outputChannels)
                .setKernelShape(new Shape(1, 1))
                .optBias(true)
                .build();
        sigmaConv.setInitializer(xavier, ai.djl.nn.Parameter.Type.WEIGHT);
        addChildBlock("sigma_conv", sigmaConv);
    }

    /**
     * Create with default architecture settings.
     */
    public static MlpConvDense createDefault(int inputChannels, int numTemporalBasis) {
        return new MlpConvDense(
                inputChannels,
                numTemporalBasis,
                128,   // lowerHiddenChannels
                64,    // upperHiddenChannels
                4,     // numLowerLayers
                2,     // numUpperLayers
                3,     // numScales
                3      // kernelSize
        );
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape inputShape = inputShapes[0];
        long batch = inputShape.get(0);
        long height = inputShape.get(2);
        long width = inputShape.get(3);

        // Initialize lower MLP
        lowerMlp.initialize(manager, dataType, inputShape);

        // Get lower MLP output channels
        int lowerOutputChannels = (lowerHiddenChannels / numScales) * numScales;

        // Initialize upper conv layers
        int prevChannels = lowerOutputChannels;
        for (int i = 0; i < numUpperLayers; i++) {
            Shape layerInputShape = new Shape(batch, prevChannels, height, width);
            upperConvLayers.get(i).initialize(manager, dataType, layerInputShape);
            prevChannels = upperHiddenChannels;
        }

        // Initialize output layers
        Shape outputInputShape = new Shape(batch, prevChannels, height, width);
        muConv.initialize(manager, dataType, outputInputShape);
        sigmaConv.initialize(manager, dataType, outputInputShape);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {

        NDArray x = inputs.singletonOrThrow();
        Shape inputShape = x.getShape();
        long batch = inputShape.get(0);
        long height = inputShape.get(2);
        long width = inputShape.get(3);

        // Apply lower MLP (convolutional feature extraction)
        NDList lowerOutput = lowerMlp.forward(parameterStore, new NDList(x), training);
        NDArray features = lowerOutput.singletonOrThrow();

        // Apply upper MLP (1x1 convolutions = per-pixel dense layers)
        for (int i = 0; i < numUpperLayers; i++) {
            NDList convOutput = upperConvLayers.get(i).forward(parameterStore, new NDList(features), training);
            features = convOutput.singletonOrThrow();
            features = LeakyRelu.apply(features, leakySlope);
        }

        // Output layers for mu and sigma
        NDList muOutput = muConv.forward(parameterStore, new NDList(features), training);
        NDArray muCoeffs = muOutput.singletonOrThrow();

        NDList sigmaOutput = sigmaConv.forward(parameterStore, new NDList(features), training);
        NDArray sigmaCoeffs = sigmaOutput.singletonOrThrow();

        // Reshape to (batch, numTemporalBasis, inputChannels, height, width)
        muCoeffs = muCoeffs.reshape(batch, numTemporalBasis, inputChannels, height, width);
        sigmaCoeffs = sigmaCoeffs.reshape(batch, numTemporalBasis, inputChannels, height, width);

        return new NDList(muCoeffs, sigmaCoeffs);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        long batch = inputShape.get(0);
        long height = inputShape.get(2);
        long width = inputShape.get(3);

        return new Shape[]{
                new Shape(batch, numTemporalBasis, inputChannels, height, width),  // mu coefficients
                new Shape(batch, numTemporalBasis, inputChannels, height, width)   // sigma coefficients
        };
    }

    public int getInputChannels() {
        return inputChannels;
    }

    public int getNumTemporalBasis() {
        return numTemporalBasis;
    }
}
