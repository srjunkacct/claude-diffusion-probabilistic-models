package com.technodrome.diffusion.data;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Data preprocessing utilities for diffusion model training.
 *
 * Implements:
 * - Z-score normalization (standardization)
 * - Uniform noise addition (dequantization)
 * - Value range transformations
 */
public class DataPreprocessor {

    private final double mean;
    private final double std;
    private final double noiseScale;

    // Cached statistics for inverse transform
    private double dataMean = 0.0;
    private double dataStd = 1.0;
    private boolean statisticsComputed = false;

    /**
     * Create a preprocessor with specified normalization parameters.
     *
     * @param mean Mean for z-score normalization
     * @param std Standard deviation for z-score normalization
     * @param noiseScale Scale of uniform noise for dequantization
     */
    public DataPreprocessor(double mean, double std, double noiseScale) {
        this.mean = mean;
        this.std = std;
        this.noiseScale = noiseScale;
    }

    /**
     * Create a preprocessor with default MNIST settings.
     * MNIST images are normalized to [0, 1], so we use typical MNIST stats.
     */
    public static DataPreprocessor createMnistPreprocessor() {
        // MNIST approximate statistics after [0,1] normalization
        return new DataPreprocessor(0.1307, 0.3081, 1.0 / 255.0);
    }

    /**
     * Create a preprocessor that will compute statistics from data.
     */
    public static DataPreprocessor createAdaptive(double noiseScale) {
        return new DataPreprocessor(0.0, 1.0, noiseScale);
    }

    /**
     * Apply preprocessing to a batch of images.
     *
     * Steps:
     * 1. Add uniform noise for dequantization (helps with discrete pixel values)
     * 2. Apply z-score normalization: (x - mean) / std
     *
     * @param images Input images, shape (batch, channels, height, width) or (batch, height, width)
     * @return Preprocessed images
     */
    public NDArray preprocess(NDArray images) {
        NDManager manager = images.getManager();

        // Ensure 4D shape: (batch, channels, height, width)
        NDArray x = ensureBatchChannelFormat(images);

        // Add uniform noise for dequantization
        if (noiseScale > 0) {
            NDArray noise = manager.randomUniform(0, (float) noiseScale, x.getShape());
            x = x.add(noise);
        }

        // Z-score normalization
        x = x.sub(mean).div(std);

        return x;
    }

    /**
     * Inverse preprocessing to recover original value range.
     *
     * @param images Preprocessed images
     * @return Images in original value range
     */
    public NDArray inversePreprocess(NDArray images) {
        // Inverse z-score: x * std + mean
        NDArray x = images.mul(std).add(mean);

        // Clip to valid range [0, 1]
        x = x.clip(0, 1);

        return x;
    }

    /**
     * Compute and store statistics from a dataset.
     *
     * @param images Dataset images
     */
    public void computeStatistics(NDArray images) {
        this.dataMean = images.mean().getFloat();
        this.dataStd = images.sub(dataMean).pow(2).mean().sqrt().getFloat();
        this.statisticsComputed = true;
    }

    /**
     * Ensure images are in (batch, channels, height, width) format.
     */
    private NDArray ensureBatchChannelFormat(NDArray images) {
        Shape shape = images.getShape();

        if (shape.dimension() == 2) {
            // Single grayscale image: (height, width) -> (1, 1, height, width)
            return images.reshape(1, 1, shape.get(0), shape.get(1));
        } else if (shape.dimension() == 3) {
            // Batch of grayscale: (batch, height, width) -> (batch, 1, height, width)
            return images.reshape(shape.get(0), 1, shape.get(1), shape.get(2));
        } else if (shape.dimension() == 4) {
            return images;
        } else {
            throw new IllegalArgumentException("Unexpected image shape: " + shape);
        }
    }

    /**
     * Add Gaussian noise to images (for testing/visualization).
     */
    public NDArray addGaussianNoise(NDArray images, double noiseStd) {
        NDManager manager = images.getManager();
        NDArray noise = manager.randomNormal(images.getShape()).mul(noiseStd);
        return images.add(noise);
    }

    /**
     * Normalize images to [0, 1] range.
     */
    public static NDArray normalizeToUnitRange(NDArray images) {
        float min = images.min().getFloat();
        float max = images.max().getFloat();
        if (max - min < 1e-8) {
            return images.zerosLike().add(0.5f);
        }
        return images.sub(min).div(max - min);
    }

    /**
     * Rescale from [-1, 1] to [0, 1].
     */
    public static NDArray rescaleFromMinusOneOne(NDArray images) {
        return images.add(1).div(2);
    }

    /**
     * Rescale from [0, 1] to [-1, 1].
     */
    public static NDArray rescaleToMinusOneOne(NDArray images) {
        return images.mul(2).sub(1);
    }

    // Getters
    public double getMean() {
        return mean;
    }

    public double getStd() {
        return std;
    }

    public double getNoiseScale() {
        return noiseScale;
    }

    public double getDataMean() {
        return dataMean;
    }

    public double getDataStd() {
        return dataStd;
    }

    public boolean isStatisticsComputed() {
        return statisticsComputed;
    }
}
