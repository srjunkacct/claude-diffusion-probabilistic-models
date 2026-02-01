package com.technodrome.diffusion.network;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Main MLP architecture for the diffusion model.
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
    private MultiLayerConvolution lowerMlp;

    // Upper MLP parameters (1x1 convolutions)
    private final java.util.List<Parameter> upperWeights;
    private final java.util.List<Parameter> upperBiases;

    // Output layer parameters
    private Parameter muWeight;
    private Parameter muBias;
    private Parameter sigmaWeight;
    private Parameter sigmaBias;

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
                lowerHiddenChannels,  // Output same as hidden for lower
                numLowerLayers,
                numScales,
                kernelSize,
                leakySlope);
        addChildBlock("lower_mlp", lowerMlp);

        // Create upper MLP parameters (1x1 convolutions = per-pixel dense)
        this.upperWeights = new java.util.ArrayList<>();
        this.upperBiases = new java.util.ArrayList<>();

        for (int i = 0; i < numUpperLayers; i++) {
            Parameter weight = addParameter(
                    Parameter.builder()
                            .setName("upper_weight" + i)
                            .setType(Parameter.Type.WEIGHT)
                            .build());
            Parameter bias = addParameter(
                    Parameter.builder()
                            .setName("upper_bias" + i)
                            .setType(Parameter.Type.BIAS)
                            .build());
            upperWeights.add(weight);
            upperBiases.add(bias);
        }

        // Output layers for mu and sigma coefficients
        this.muWeight = addParameter(
                Parameter.builder()
                        .setName("mu_weight")
                        .setType(Parameter.Type.WEIGHT)
                        .build());
        this.muBias = addParameter(
                Parameter.builder()
                        .setName("mu_bias")
                        .setType(Parameter.Type.BIAS)
                        .build());
        this.sigmaWeight = addParameter(
                Parameter.builder()
                        .setName("sigma_weight")
                        .setType(Parameter.Type.WEIGHT)
                        .build());
        this.sigmaBias = addParameter(
                Parameter.builder()
                        .setName("sigma_bias")
                        .setType(Parameter.Type.BIAS)
                        .build());
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
    public void prepare(Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        long height = inputShape.get(2);
        long width = inputShape.get(3);

        // Prepare lower MLP
        lowerMlp.prepare(inputShapes);

        // Upper MLP dimensions
        int lowerOutputChannels = lowerHiddenChannels;
        int prevChannels = lowerOutputChannels;

        for (int i = 0; i < numUpperLayers; i++) {
            int currOutputChannels = upperHiddenChannels;
            upperWeights.get(i).setShape(new Shape(currOutputChannels, prevChannels, 1, 1));
            upperBiases.get(i).setShape(new Shape(currOutputChannels));
            prevChannels = currOutputChannels;
        }

        // Output layer dimensions
        // Output: numTemporalBasis coefficients per channel for both mu and sigma
        int muOutputChannels = inputChannels * numTemporalBasis;
        int sigmaOutputChannels = inputChannels * numTemporalBasis;

        muWeight.setShape(new Shape(muOutputChannels, prevChannels, 1, 1));
        muBias.setShape(new Shape(muOutputChannels));
        sigmaWeight.setShape(new Shape(sigmaOutputChannels, prevChannels, 1, 1));
        sigmaBias.setShape(new Shape(sigmaOutputChannels));
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {

        NDArray x = inputs.singletonOrThrow();
        NDManager manager = x.getManager();
        Shape inputShape = x.getShape();
        long batch = inputShape.get(0);
        long height = inputShape.get(2);
        long width = inputShape.get(3);

        // Apply lower MLP (convolutional feature extraction)
        NDList lowerOutput = lowerMlp.forward(parameterStore, new NDList(x), training);
        NDArray features = lowerOutput.singletonOrThrow();

        // Apply upper MLP (1x1 convolutions = per-pixel dense layers)
        for (int i = 0; i < numUpperLayers; i++) {
            NDArray weight = parameterStore.getValue(upperWeights.get(i), x.getDevice(), training);
            NDArray bias = parameterStore.getValue(upperBiases.get(i), x.getDevice(), training);

            features = conv1x1(features, weight, bias);
            features = LeakyRelu.apply(features, leakySlope);
        }

        // Output layers for mu and sigma
        NDArray muWeightVal = parameterStore.getValue(muWeight, x.getDevice(), training);
        NDArray muBiasVal = parameterStore.getValue(muBias, x.getDevice(), training);
        NDArray muCoeffs = conv1x1(features, muWeightVal, muBiasVal);

        NDArray sigmaWeightVal = parameterStore.getValue(sigmaWeight, x.getDevice(), training);
        NDArray sigmaBiasVal = parameterStore.getValue(sigmaBias, x.getDevice(), training);
        NDArray sigmaCoeffs = conv1x1(features, sigmaWeightVal, sigmaBiasVal);

        // Reshape to (batch, numTemporalBasis, inputChannels, height, width)
        muCoeffs = muCoeffs.reshape(batch, numTemporalBasis, inputChannels, height, width);
        sigmaCoeffs = sigmaCoeffs.reshape(batch, numTemporalBasis, inputChannels, height, width);

        return new NDList(muCoeffs, sigmaCoeffs);
    }

    private NDArray conv1x1(NDArray input, NDArray weight, NDArray bias) {
        // 1x1 convolution (per-pixel linear transformation)
        // Implemented as matrix multiplication across spatial dimensions
        Shape inputShape = input.getShape();
        long batch = inputShape.get(0);
        long inChannels = inputShape.get(1);
        long height = inputShape.get(2);
        long width = inputShape.get(3);

        Shape weightShape = weight.getShape();
        long outChannels = weightShape.get(0);

        // Reshape input to (batch, inChannels, H*W)
        NDArray flatInput = input.reshape(batch, inChannels, height * width);

        // Weight is (outChannels, inChannels, 1, 1) -> (outChannels, inChannels)
        NDArray flatWeight = weight.reshape(outChannels, inChannels);

        // Compute: output = weight @ input for each batch and spatial position
        // flatInput: (batch, inChannels, H*W)
        // flatWeight: (outChannels, inChannels)
        // We need: (batch, outChannels, H*W) = flatWeight @ flatInput

        NDArray output = flatWeight.matMul(flatInput);  // (outChannels, H*W) for each batch element
        output = output.reshape(batch, outChannels, height, width);

        // Add bias
        output = output.add(bias.reshape(1, outChannels, 1, 1));

        return output;
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
