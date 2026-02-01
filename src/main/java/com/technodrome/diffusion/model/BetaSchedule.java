package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Beta noise variance schedule for the forward diffusion process.
 *
 * Controls how noise is added at each timestep during forward diffusion.
 * The schedule starts with small beta values and increases towards 1.0,
 * gradually corrupting the input towards pure Gaussian noise.
 */
public class BetaSchedule {

    private final int trajectoryLength;
    private final double step1Beta;
    private final double minBeta;

    /**
     * Create a beta schedule.
     *
     * @param trajectoryLength Number of diffusion timesteps
     * @param step1Beta Beta value at the first step
     * @param minBeta Minimum beta value (for numerical stability)
     */
    public BetaSchedule(int trajectoryLength, double step1Beta, double minBeta) {
        this.trajectoryLength = trajectoryLength;
        this.step1Beta = step1Beta;
        this.minBeta = minBeta;
    }

    /**
     * Create a beta schedule with default minimum beta.
     */
    public BetaSchedule(int trajectoryLength, double step1Beta) {
        this(trajectoryLength, step1Beta, 1e-5);
    }

    /**
     * Generate the beta array for all timesteps.
     *
     * Beta values are linearly interpolated from step1Beta at t=1
     * to approximately 1.0 at t=T (trajectory_length).
     *
     * @param manager NDManager for array creation
     * @return NDArray of shape (trajectoryLength,) containing beta_t values
     */
    public NDArray generateBetaArray(NDManager manager) {
        float[] betas = new float[trajectoryLength];

        // Linear interpolation from step1Beta to near 1.0
        // The formula ensures beta_T approaches 1.0 as T increases
        for (int t = 0; t < trajectoryLength; t++) {
            double tNorm = (double) (t + 1) / trajectoryLength;
            // Linear schedule: beta_t = step1_beta + t/T * (1 - step1_beta)
            double beta = step1Beta + tNorm * (1.0 - step1Beta);
            // Ensure minimum beta for numerical stability
            betas[t] = (float) Math.max(beta, minBeta);
        }

        return manager.create(betas, new Shape(trajectoryLength));
    }

    /**
     * Generate alpha values: alpha_t = 1 - beta_t
     */
    public NDArray generateAlphaArray(NDManager manager) {
        NDArray betas = generateBetaArray(manager);
        return betas.neg().add(1.0);
    }

    /**
     * Generate cumulative alpha product: alpha_bar_t = prod(alpha_1, ..., alpha_t)
     * This represents how much of the original signal remains at step t.
     */
    public NDArray generateAlphaCumprod(NDManager manager) {
        NDArray alphas = generateAlphaArray(manager);
        float[] alphaData = alphas.toFloatArray();
        float[] alphaCumprod = new float[trajectoryLength];

        alphaCumprod[0] = alphaData[0];
        for (int t = 1; t < trajectoryLength; t++) {
            alphaCumprod[t] = alphaCumprod[t - 1] * alphaData[t];
        }

        return manager.create(alphaCumprod, new Shape(trajectoryLength));
    }

    /**
     * Get the square root of cumulative alpha product.
     * Used as the coefficient for the original signal in forward diffusion:
     * x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
     */
    public NDArray generateSqrtAlphaCumprod(NDManager manager) {
        return generateAlphaCumprod(manager).sqrt();
    }

    /**
     * Get the square root of (1 - cumulative alpha product).
     * Used as the coefficient for noise in forward diffusion.
     */
    public NDArray generateSqrtOneMinusAlphaCumprod(NDManager manager) {
        NDArray alphaCumprod = generateAlphaCumprod(manager);
        return alphaCumprod.neg().add(1.0).sqrt();
    }

    /**
     * Get beta value at specific timestep (0-indexed).
     */
    public double getBeta(int t) {
        double tNorm = (double) (t + 1) / trajectoryLength;
        double beta = step1Beta + tNorm * (1.0 - step1Beta);
        return Math.max(beta, minBeta);
    }

    /**
     * Get the posterior variance for the reverse process.
     * beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
     *
     * This is the variance of q(x_{t-1} | x_t, x_0)
     */
    public NDArray generatePosteriorVariance(NDManager manager) {
        NDArray betas = generateBetaArray(manager);
        NDArray alphaCumprod = generateAlphaCumprod(manager);

        float[] betaData = betas.toFloatArray();
        float[] alphaCumprodData = alphaCumprod.toFloatArray();
        float[] posteriorVar = new float[trajectoryLength];

        // At t=0, posterior variance is not well-defined, set to beta_0
        posteriorVar[0] = betaData[0];

        for (int t = 1; t < trajectoryLength; t++) {
            double alphaCumprodPrev = alphaCumprodData[t - 1];
            double alphaCumprodCurr = alphaCumprodData[t];
            double beta = betaData[t];

            posteriorVar[t] = (float) (beta * (1.0 - alphaCumprodPrev) / (1.0 - alphaCumprodCurr));
        }

        return manager.create(posteriorVar, new Shape(trajectoryLength));
    }

    public int getTrajectoryLength() {
        return trajectoryLength;
    }

    public double getStep1Beta() {
        return step1Beta;
    }

    public double getMinBeta() {
        return minBeta;
    }
}
