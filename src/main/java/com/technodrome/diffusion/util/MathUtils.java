package com.technodrome.diffusion.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Mathematical utility functions for diffusion models.
 * Includes logit, sigmoid, softplus, and related operations.
 */
public final class MathUtils {

    private static final double EPSILON = 1e-7;
    private static final double SOFTPLUS_THRESHOLD = 20.0;

    private MathUtils() {
        // Utility class
    }

    /**
     * Logit function: log(x / (1 - x))
     * Inverse of sigmoid.
     */
    public static NDArray logit(NDArray x) {
        // Clip to avoid log(0) and division by zero
        NDArray clipped = x.clip(EPSILON, 1.0 - EPSILON);
        return clipped.div(clipped.neg().add(1.0)).log();
    }

    /**
     * Sigmoid function: 1 / (1 + exp(-x))
     */
    public static NDArray sigmoid(NDArray x) {
        return x.neg().exp().add(1.0).pow(-1);
    }

    /**
     * Softplus function: log(1 + exp(x))
     * Uses a numerically stable implementation that avoids overflow.
     */
    public static NDArray softplus(NDArray x) {
        // Numerically stable: softplus(x) = max(0, x) + log(1 + exp(-|x|))
        // This avoids overflow from exp(large_positive)
        NDArray absX = x.abs();
        NDArray negAbsX = absX.neg();
        // log(1 + exp(-|x|)) is always safe since exp(-|x|) <= 1
        NDArray logTerm = negAbsX.exp().add(1.0).log();
        // max(0, x) + log(1 + exp(-|x|))
        NDArray result = x.maximum(0).add(logTerm);
        return result;
    }

    /**
     * Inverse softplus: log(exp(x) - 1)
     */
    public static NDArray inverseSoftplus(NDArray x) {
        // For large x, inverse_softplus(x) ≈ x
        NDArray expX = x.exp();
        NDArray result = expX.sub(1.0).log();
        NDArray mask = x.gt(SOFTPLUS_THRESHOLD);
        return result.mul(mask.logicalNot().toType(result.getDataType(), false))
                .add(x.mul(mask.toType(x.getDataType(), false)));
    }

    /**
     * Log-sum-exp trick for numerical stability.
     * Computes log(sum(exp(x))) along the specified axis.
     */
    public static NDArray logSumExp(NDArray x, int axis) {
        NDArray maxX = x.max(new int[]{axis}, true);
        return x.sub(maxX).exp().sum(new int[]{axis}, true).log().add(maxX);
    }

    /**
     * Gaussian log probability density.
     * log N(x; mu, sigma) = -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
     */
    public static NDArray gaussianLogPdf(NDArray x, NDArray mu, NDArray sigma) {
        double logTwoPi = Math.log(2.0 * Math.PI);
        NDArray zScore = x.sub(mu).div(sigma);
        return sigma.log().neg()
                .sub(0.5 * logTwoPi)
                .sub(zScore.pow(2).mul(0.5));
    }

    /**
     * KL divergence between two Gaussians.
     * KL(N(mu1, sigma1) || N(mu2, sigma2))
     *
     */
    public static NDArray gaussianKL(NDArray mu1, NDArray sigma1, NDArray mu2, NDArray sigma2) {
        NDArray logRatio = sigma2.log().sub(sigma1.log());
        return logRatio
                .add( sigma1.pow(2).add( mu1.sub(mu2).pow(2) ).div( sigma2.pow(2).mul(2 ) ) )
                                .sub(0.5 );
    }

    /**
     * Standard Gaussian log probability.
     * log N(x; 0, 1)
     */
    public static NDArray standardGaussianLogPdf(NDArray x) {
        double logTwoPi = Math.log(2.0 * Math.PI);
        return x.pow(2).mul(-0.5).sub(0.5 * logTwoPi);
    }

    /**
     * Entropy of a Gaussian with given sigma.
     * H(N(mu, sigma)) = 0.5 * log(2 * pi * e * sigma^2)
     */
    public static NDArray gaussianEntropy(NDArray sigma) {
        double logTwoPiE = Math.log(2.0 * Math.PI * Math.E);
        return sigma.log().mul(2.0).add(logTwoPiE).mul(0.5);
    }

    /**
     * Create a linear space from start to end with n steps.
     */
    public static NDArray linspace(NDManager manager, double start, double end, int steps) {
        return manager.linspace((float) start, (float) end, steps);
    }

    /**
     * Create a meshgrid of coordinates.
     */
    public static NDArray[] meshgrid(NDManager manager, int height, int width) {
        NDArray y = manager.arange(height).reshape(height, 1).tile(1, width);
        NDArray x = manager.arange(width).reshape(1, width).tile(height, 1);
        return new NDArray[]{y, x};
    }

    /**
     * Compute cumulative product along an axis.
     */
    public static NDArray cumprod(NDArray x) {
        long length = x.size();
        NDArray result = x.getManager().zeros(x.getShape(), x.getDataType());
        float[] data = x.toFloatArray();
        float[] cumData = new float[(int) length];
        cumData[0] = data[0];
        for (int i = 1; i < length; i++) {
            cumData[i] = cumData[i - 1] * data[i];
        }
        return x.getManager().create(cumData, x.getShape());
    }

    /**
     * Mean pooling for downsampling.
     * Reduces spatial dimensions by factor of 2.
     */
    public static NDArray meanPool2x2(NDArray x) {
        // x shape: (batch, channels, height, width)
        Shape shape = x.getShape();
        long batch = shape.get(0);
        long channels = shape.get(1);
        long height = shape.get(2);
        long width = shape.get(3);

        // Reshape to (batch, channels, height/2, 2, width/2, 2)
        NDArray reshaped = x.reshape(batch, channels, height / 2, 2, width / 2, 2);
        // Mean over axes 5 then 3 (chain for PyTorch compatibility)
        return reshaped.mean(new int[]{5}).mean(new int[]{3});
    }

    /**
     * Upsample by 2x using nearest neighbor (replication).
     */
    public static NDArray upsample2x(NDArray x) {
        // x shape: (batch, channels, height, width)
        Shape shape = x.getShape();
        long batch = shape.get(0);
        long channels = shape.get(1);
        long height = shape.get(2);
        long width = shape.get(3);

        // Repeat along height and width dimensions
        return x.repeat(2, 2).repeat(3, 2);
    }
}
