package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Temporal basis functions for time-step dependent predictions.
 *
 * Uses Gaussian bump basis functions centered at different points in time
 * to allow the network to output time-varying predictions for mu and sigma
 * of the reverse diffusion process.
 */
public class TemporalBasis {

    private final int trajectoryLength;
    private final int numBasisFunctions;

    /**
     * Create temporal basis generator.
     *
     * @param trajectoryLength Number of diffusion timesteps
     * @param numBasisFunctions Number of Gaussian bump basis functions
     */
    public TemporalBasis(int trajectoryLength, int numBasisFunctions) {
        this.trajectoryLength = trajectoryLength;
        this.numBasisFunctions = numBasisFunctions;
    }

    /**
     * Generate the temporal basis matrix.
     *
     * Creates a matrix of shape (trajectoryLength, numBasisFunctions) where
     * each column is a Gaussian bump centered at a different timestep.
     *
     * @param manager NDManager for array creation
     * @return NDArray of shape (trajectoryLength, numBasisFunctions)
     */
    public NDArray generate(NDManager manager) {
        float[][] basis = new float[trajectoryLength][numBasisFunctions];

        // Centers of Gaussian bumps, evenly spaced
        double[] centers = new double[numBasisFunctions];
        for (int i = 0; i < numBasisFunctions; i++) {
            centers[i] = (double) i / (numBasisFunctions - 1);
        }

        // Width of Gaussian bumps - overlapping for smooth interpolation
        double sigma = 1.0 / (numBasisFunctions - 1);

        for (int t = 0; t < trajectoryLength; t++) {
            double tNorm = (double) t / (trajectoryLength - 1);

            for (int b = 0; b < numBasisFunctions; b++) {
                double diff = tNorm - centers[b];
                double value = Math.exp(-0.5 * (diff * diff) / (sigma * sigma));
                basis[t][b] = (float) value;
            }
        }

        // Normalize each row so basis functions sum to 1 at each timestep
        // This ensures stable interpolation
        float[][] normalizedBasis = new float[trajectoryLength][numBasisFunctions];
        for (int t = 0; t < trajectoryLength; t++) {
            float sum = 0;
            for (int b = 0; b < numBasisFunctions; b++) {
                sum += basis[t][b];
            }
            for (int b = 0; b < numBasisFunctions; b++) {
                normalizedBasis[t][b] = basis[t][b] / sum;
            }
        }

        // Flatten to 1D then reshape
        float[] flatBasis = new float[trajectoryLength * numBasisFunctions];
        for (int t = 0; t < trajectoryLength; t++) {
            for (int b = 0; b < numBasisFunctions; b++) {
                flatBasis[t * numBasisFunctions + b] = normalizedBasis[t][b];
            }
        }

        return manager.create(flatBasis, new Shape(trajectoryLength, numBasisFunctions));
    }

    /**
     * Generate unnormalized basis for cases where raw Gaussians are needed.
     */
    public NDArray generateUnnormalized(NDManager manager) {
        float[][] basis = new float[trajectoryLength][numBasisFunctions];

        double[] centers = new double[numBasisFunctions];
        for (int i = 0; i < numBasisFunctions; i++) {
            centers[i] = (double) i / (numBasisFunctions - 1);
        }

        double sigma = 1.0 / (numBasisFunctions - 1);

        for (int t = 0; t < trajectoryLength; t++) {
            double tNorm = (double) t / (trajectoryLength - 1);
            for (int b = 0; b < numBasisFunctions; b++) {
                double diff = tNorm - centers[b];
                double value = Math.exp(-0.5 * (diff * diff) / (sigma * sigma));
                basis[t][b] = (float) value;
            }
        }

        float[] flatBasis = new float[trajectoryLength * numBasisFunctions];
        for (int t = 0; t < trajectoryLength; t++) {
            for (int b = 0; b < numBasisFunctions; b++) {
                flatBasis[t * numBasisFunctions + b] = basis[t][b];
            }
        }

        return manager.create(flatBasis, new Shape(trajectoryLength, numBasisFunctions));
    }

    /**
     * Get basis values at a specific timestep.
     *
     * @param manager NDManager for array creation
     * @param t Timestep index
     * @return NDArray of shape (numBasisFunctions,)
     */
    public NDArray getBasisAtTime(NDManager manager, int t) {
        NDArray fullBasis = generate(manager);
        return fullBasis.get(t);
    }

    /**
     * Get basis values for multiple timesteps (batched).
     *
     * @param manager NDManager for array creation
     * @param timesteps Array of timestep indices
     * @return NDArray of shape (len(timesteps), numBasisFunctions)
     */
    public NDArray getBasisAtTimes(NDManager manager, int[] timesteps) {
        NDArray fullBasis = generate(manager);
        int[] indices = new int[timesteps.length];
        System.arraycopy(timesteps, 0, indices, 0, timesteps.length);

        NDArray indicesArray = manager.create(indices);
        // Use gather to select rows
        float[][] result = new float[timesteps.length][numBasisFunctions];
        float[] basisData = fullBasis.toFloatArray();

        for (int i = 0; i < timesteps.length; i++) {
            int t = timesteps[i];
            for (int b = 0; b < numBasisFunctions; b++) {
                result[i][b] = basisData[t * numBasisFunctions + b];
            }
        }

        float[] flatResult = new float[timesteps.length * numBasisFunctions];
        for (int i = 0; i < timesteps.length; i++) {
            for (int b = 0; b < numBasisFunctions; b++) {
                flatResult[i * numBasisFunctions + b] = result[i][b];
            }
        }

        return manager.create(flatResult, new Shape(timesteps.length, numBasisFunctions));
    }

    /**
     * Apply temporal basis to coefficients to get time-specific values.
     *
     * @param coefficients Network output of shape (batch, numBasisFunctions, ...)
     * @param timesteps Timesteps for each batch item
     * @return Values at specified timesteps, shape (batch, ...)
     */
    public NDArray apply(NDArray coefficients, NDArray basis) {
        // coefficients: (batch, numBasis, channels, height, width) or similar
        // basis: (batch, numBasis) or (numBasis,)
        // Output: weighted sum over numBasis dimension

        return coefficients.mul(basis).sum(new int[]{1});
    }

    public int getTrajectoryLength() {
        return trajectoryLength;
    }

    public int getNumBasisFunctions() {
        return numBasisFunctions;
    }
}
