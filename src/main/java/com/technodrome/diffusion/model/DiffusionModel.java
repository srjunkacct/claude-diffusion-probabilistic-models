package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import com.technodrome.diffusion.network.MlpConvDense;
import com.technodrome.diffusion.util.MathUtils;

/**
 * Core diffusion probabilistic model.
 *
 * Implements:
 * - Forward diffusion process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
 * - Reverse process learning: p(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_theta(x_t, t))
 * - Training objective: Negative log-likelihood bound via KL divergence
 */
public class DiffusionModel extends AbstractBlock {

    private static final byte VERSION = 1;
    private static final double MIN_SIGMA = 1e-5;

    private final int trajectoryLength;
    private final int numTemporalBasis;
    private final int imageChannels;
    private final int imageHeight;
    private final int imageWidth;

    private final BetaSchedule betaSchedule;
    private final TemporalBasis temporalBasis;
    private final MlpConvDense network;

    // Cached diffusion parameters
    private NDArray betas;
    private NDArray alphas;
    private NDArray alphaCumprod;
    private NDArray sqrtAlphaCumprod;
    private NDArray sqrtOneMinusAlphaCumprod;
    private NDArray temporalBasisMatrix;

    /**
     * Create a diffusion model.
     *
     * @param trajectoryLength Number of diffusion timesteps (T)
     * @param numTemporalBasis Number of temporal basis functions
     * @param imageChannels Number of image channels
     * @param imageHeight Image height
     * @param imageWidth Image width
     * @param step1Beta Beta value at t=1
     */
    public DiffusionModel(int trajectoryLength, int numTemporalBasis,
                          int imageChannels, int imageHeight, int imageWidth,
                          double step1Beta) {
        super(VERSION);
        this.trajectoryLength = trajectoryLength;
        this.numTemporalBasis = numTemporalBasis;
        this.imageChannels = imageChannels;
        this.imageHeight = imageHeight;
        this.imageWidth = imageWidth;

        this.betaSchedule = new BetaSchedule(trajectoryLength, step1Beta);
        this.temporalBasis = new TemporalBasis(trajectoryLength, numTemporalBasis);

        // Create the neural network for predicting mu and sigma
        this.network = MlpConvDense.createDefault(imageChannels, numTemporalBasis);
        addChildBlock("network", network);
    }

    /**
     * Create with default MNIST settings.
     */
    public static DiffusionModel createMnistModel(int trajectoryLength, int numTemporalBasis) {
        return new DiffusionModel(
                trajectoryLength,
                numTemporalBasis,
                1,    // channels
                28,   // height
                28,   // width
                1e-4  // step1Beta
        );
    }

    @Override
    public void prepare(Shape[] inputShapes) {
        network.prepare(inputShapes);
    }

    /**
     * Initialize cached diffusion parameters.
     */
    public void initializeDiffusionParams(NDManager manager) {
        this.betas = betaSchedule.generateBetaArray(manager);
        this.alphas = betaSchedule.generateAlphaArray(manager);
        this.alphaCumprod = betaSchedule.generateAlphaCumprod(manager);
        this.sqrtAlphaCumprod = betaSchedule.generateSqrtAlphaCumprod(manager);
        this.sqrtOneMinusAlphaCumprod = betaSchedule.generateSqrtOneMinusAlphaCumprod(manager);
        this.temporalBasisMatrix = temporalBasis.generate(manager);
    }

    /**
     * Forward diffusion: sample x_t given x_0 and timestep t.
     *
     * q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
     *
     * @param x0 Original images, shape (batch, channels, height, width)
     * @param t Timesteps for each batch item, shape (batch,)
     * @param noise Optional noise to use (for reproducibility)
     * @return Noisy images x_t and the noise used
     */
    public NDList getForwardDiffusionSample(NDArray x0, NDArray t, NDArray noise) {
        NDManager manager = x0.getManager();
        Shape shape = x0.getShape();

        if (noise == null) {
            noise = manager.randomNormal(shape);
        }

        // Get alpha_bar_t for each batch item
        float[] sqrtAlphaData = sqrtAlphaCumprod.toFloatArray();
        float[] sqrtOneMinusAlphaData = sqrtOneMinusAlphaCumprod.toFloatArray();
        int[] timesteps = t.toIntArray();

        // Build coefficient arrays for each batch item
        float[] sqrtAlphaCoeffs = new float[timesteps.length];
        float[] sqrtOneMinusAlphaCoeffs = new float[timesteps.length];

        for (int i = 0; i < timesteps.length; i++) {
            int idx = Math.min(timesteps[i], trajectoryLength - 1);
            sqrtAlphaCoeffs[i] = sqrtAlphaData[idx];
            sqrtOneMinusAlphaCoeffs[i] = sqrtOneMinusAlphaData[idx];
        }

        // Create coefficient tensors with proper broadcast shape
        NDArray sqrtAlphaT = manager.create(sqrtAlphaCoeffs).reshape(-1, 1, 1, 1);
        NDArray sqrtOneMinusAlphaT = manager.create(sqrtOneMinusAlphaCoeffs).reshape(-1, 1, 1, 1);

        // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        NDArray xt = x0.mul(sqrtAlphaT).add(noise.mul(sqrtOneMinusAlphaT));

        return new NDList(xt, noise);
    }

    /**
     * Get mu and sigma predictions from the network.
     *
     * @param xt Noisy images at time t
     * @param t Timesteps
     * @param parameterStore Parameter store for network weights
     * @param training Whether in training mode
     * @return NDList containing [mu, sigma]
     */
    public NDList getMuSigma(NDArray xt, NDArray t,
                              ParameterStore parameterStore, boolean training) {
        NDManager manager = xt.getManager();

        // Get temporal basis for each timestep
        int[] timesteps = t.toIntArray();
        NDArray basis = temporalBasis.getBasisAtTimes(manager, timesteps);
        // basis shape: (batch, numTemporalBasis)

        // Get network predictions
        NDList networkOutput = network.forward(parameterStore, new NDList(xt), training);
        NDArray muCoeffs = networkOutput.get(0);     // (batch, numTemporalBasis, channels, height, width)
        NDArray sigmaCoeffs = networkOutput.get(1);  // (batch, numTemporalBasis, channels, height, width)

        // Apply temporal basis to get time-specific predictions
        // Reshape basis for broadcasting: (batch, numTemporalBasis, 1, 1, 1)
        NDArray basisExpanded = basis.reshape(basis.getShape().get(0), numTemporalBasis, 1, 1, 1);

        // Weighted sum over temporal basis dimension
        NDArray mu = muCoeffs.mul(basisExpanded).sum(new int[]{1});  // (batch, channels, height, width)
        NDArray sigmaRaw = sigmaCoeffs.mul(basisExpanded).sum(new int[]{1});

        // Apply softplus to sigma to ensure positivity, add minimum for stability
        NDArray sigma = MathUtils.softplus(sigmaRaw).add(MIN_SIGMA);

        return new NDList(mu, sigma);
    }

    /**
     * Compute the negative lower bound (loss) for training.
     *
     * Based on KL divergence between:
     * - q(x_{t-1} | x_t, x_0): the true posterior (given we know x_0)
     * - p(x_{t-1} | x_t): our learned reverse process
     *
     * @param x0 Original images
     * @param parameterStore Parameter store
     * @param training Training mode
     * @return Scalar loss value
     */
    public NDArray getNegLBound(NDArray x0, ParameterStore parameterStore, boolean training) {
        NDManager manager = x0.getManager();
        long batchSize = x0.getShape().get(0);

        // Sample random timesteps for each batch item
        NDArray t = manager.randomUniform(0, trajectoryLength, new Shape(batchSize))
                .floor().toType(DataType.INT32, false);

        // Forward diffusion: get x_t
        NDList forwardResult = getForwardDiffusionSample(x0, t, null);
        NDArray xt = forwardResult.get(0);
        NDArray epsilon = forwardResult.get(1);

        // Get network predictions for mu and sigma
        NDList muSigma = getMuSigma(xt, t, parameterStore, training);
        NDArray muPred = muSigma.get(0);
        NDArray sigmaPred = muSigma.get(1);

        // Compute the true posterior mean
        // mu_q = (sqrt(alpha_bar_{t-1}) * beta_t * x_0 + sqrt(alpha_t) * (1 - alpha_bar_{t-1}) * x_t) / (1 - alpha_bar_t)
        int[] timesteps = t.toIntArray();
        NDArray muTrue = computePosteriorMean(x0, xt, timesteps, manager);
        NDArray sigmaTrue = computePosteriorStd(timesteps, manager);

        // KL divergence: D_KL(q || p) for each pixel
        NDArray kl = MathUtils.gaussianKL(muTrue, sigmaTrue, muPred, sigmaPred);

        // For t=0, use reconstruction loss instead
        NDArray isT0 = t.eq(0).toType(DataType.FLOAT32, false).reshape(-1, 1, 1, 1);
        NDArray reconLoss = x0.sub(muPred).pow(2).div(sigmaPred.pow(2).mul(2))
                .add(sigmaPred.log());
        NDArray loss = kl.mul(isT0.neg().add(1)).add(reconLoss.mul(isT0));

        // Sum over pixels, mean over batch
        return loss.sum(new int[]{1, 2, 3}).mean();
    }

    private NDArray computePosteriorMean(NDArray x0, NDArray xt, int[] timesteps, NDManager manager) {
        float[] betaData = betas.toFloatArray();
        float[] alphaCumprodData = alphaCumprod.toFloatArray();
        float[] alphaData = alphas.toFloatArray();

        long batchSize = x0.getShape().get(0);
        float[] coeff1 = new float[(int) batchSize];  // coefficient for x_0
        float[] coeff2 = new float[(int) batchSize];  // coefficient for x_t

        for (int i = 0; i < batchSize; i++) {
            int t = timesteps[i];
            if (t == 0) {
                coeff1[i] = 1.0f;
                coeff2[i] = 0.0f;
            } else {
                float alphaCumprodT = alphaCumprodData[t];
                float alphaCumprodPrev = alphaCumprodData[t - 1];
                float betaT = betaData[t];
                float alphaT = alphaData[t];

                // mu_q = (sqrt(alpha_bar_{t-1}) * beta_t * x_0 + sqrt(alpha_t) * (1 - alpha_bar_{t-1}) * x_t) / (1 - alpha_bar_t)
                float denom = 1.0f - alphaCumprodT;
                coeff1[i] = (float) (Math.sqrt(alphaCumprodPrev) * betaT / denom);
                coeff2[i] = (float) (Math.sqrt(alphaT) * (1.0 - alphaCumprodPrev) / denom);
            }
        }

        NDArray c1 = manager.create(coeff1).reshape(-1, 1, 1, 1);
        NDArray c2 = manager.create(coeff2).reshape(-1, 1, 1, 1);

        return x0.mul(c1).add(xt.mul(c2));
    }

    private NDArray computePosteriorStd(int[] timesteps, NDManager manager) {
        float[] betaData = betas.toFloatArray();
        float[] alphaCumprodData = alphaCumprod.toFloatArray();

        float[] posteriorVar = new float[timesteps.length];
        for (int i = 0; i < timesteps.length; i++) {
            int t = timesteps[i];
            if (t == 0) {
                posteriorVar[i] = (float) MIN_SIGMA;
            } else {
                float alphaCumprodT = alphaCumprodData[t];
                float alphaCumprodPrev = alphaCumprodData[t - 1];
                float betaT = betaData[t];
                // sigma_q^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
                posteriorVar[i] = betaT * (1.0f - alphaCumprodPrev) / (1.0f - alphaCumprodT);
                posteriorVar[i] = Math.max(posteriorVar[i], (float) MIN_SIGMA);
            }
        }

        NDArray std = manager.create(posteriorVar).sqrt().reshape(-1, 1, 1, 1);
        return std;
    }

    /**
     * Compute training cost (wrapper for loss).
     */
    public NDArray cost(NDArray x0, ParameterStore parameterStore, boolean training) {
        return getNegLBound(x0, parameterStore, training);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray x0 = inputs.singletonOrThrow();
        NDArray loss = getNegLBound(x0, parameterStore, training);
        return new NDList(loss);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape()};  // Scalar loss
    }

    // Getters
    public int getTrajectoryLength() {
        return trajectoryLength;
    }

    public int getNumTemporalBasis() {
        return numTemporalBasis;
    }

    public BetaSchedule getBetaSchedule() {
        return betaSchedule;
    }

    public TemporalBasis getTemporalBasis() {
        return temporalBasis;
    }

    public MlpConvDense getNetwork() {
        return network;
    }

    public NDArray getBetas() {
        return betas;
    }

    public NDArray getAlphaCumprod() {
        return alphaCumprod;
    }

    public NDArray getSqrtAlphaCumprod() {
        return sqrtAlphaCumprod;
    }

    public NDArray getSqrtOneMinusAlphaCumprod() {
        return sqrtOneMinusAlphaCumprod;
    }

    public NDArray getTemporalBasisMatrix() {
        return temporalBasisMatrix;
    }
}
