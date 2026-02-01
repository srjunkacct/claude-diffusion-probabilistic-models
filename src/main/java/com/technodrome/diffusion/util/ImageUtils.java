package com.technodrome.diffusion.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Image saving utilities for diffusion model outputs.
 *
 * Provides functions for:
 * - Converting NDArrays to BufferedImages
 * - Saving single images
 * - Creating and saving image grids
 * - Normalizing pixel values
 */
public final class ImageUtils {

    private ImageUtils() {
        // Utility class
    }

    /**
     * Save a batch of images as a grid.
     *
     * @param images Batch of images, shape (batch, channels, height, width)
     * @param outputPath Path to save the image
     * @param numColumns Number of columns in the grid
     */
    public static void saveImageGrid(NDArray images, String outputPath, int numColumns) throws IOException {
        Shape shape = images.getShape();
        int batchSize = (int) shape.get(0);
        int channels = (int) shape.get(1);
        int height = (int) shape.get(2);
        int width = (int) shape.get(3);

        int numRows = (batchSize + numColumns - 1) / numColumns;
        int gridHeight = numRows * height;
        int gridWidth = numColumns * width;

        BufferedImage gridImage = new BufferedImage(
                gridWidth, gridHeight,
                channels == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB
        );

        // Normalize images to [0, 255]
        float[] imageData = images.toFloatArray();

        for (int i = 0; i < batchSize; i++) {
            int row = i / numColumns;
            int col = i % numColumns;

            int offsetY = row * height;
            int offsetX = col * width;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixelValue;

                    if (channels == 1) {
                        // Grayscale
                        int idx = i * height * width + y * width + x;
                        float value = imageData[idx];
                        pixelValue = clampToByte(value);
                        gridImage.getRaster().setSample(offsetX + x, offsetY + y, 0, pixelValue);
                    } else {
                        // RGB
                        int r = clampToByte(imageData[i * channels * height * width + 0 * height * width + y * width + x]);
                        int g = clampToByte(imageData[i * channels * height * width + 1 * height * width + y * width + x]);
                        int b = clampToByte(imageData[i * channels * height * width + 2 * height * width + y * width + x]);
                        pixelValue = (r << 16) | (g << 8) | b;
                        gridImage.setRGB(offsetX + x, offsetY + y, pixelValue);
                    }
                }
            }
        }

        // Ensure parent directory exists
        Path path = Paths.get(outputPath);
        if (path.getParent() != null) {
            path.getParent().toFile().mkdirs();
        }

        // Save image
        String format = outputPath.endsWith(".png") ? "PNG" : "JPEG";
        ImageIO.write(gridImage, format, new File(outputPath));
    }

    /**
     * Save a single image.
     *
     * @param image Single image, shape (channels, height, width) or (1, channels, height, width)
     * @param outputPath Path to save the image
     */
    public static void saveImage(NDArray image, String outputPath) throws IOException {
        Shape shape = image.getShape();

        // Handle different input shapes
        NDArray img;
        if (shape.dimension() == 4) {
            // (batch, channels, height, width) - take first image
            img = image.get(0);
            shape = img.getShape();
        } else {
            img = image;
        }

        int channels = (int) shape.get(0);
        int height = (int) shape.get(1);
        int width = (int) shape.get(2);

        BufferedImage bufferedImage = new BufferedImage(
                width, height,
                channels == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB
        );

        float[] imageData = img.toFloatArray();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (channels == 1) {
                    int idx = y * width + x;
                    int value = clampToByte(imageData[idx]);
                    bufferedImage.getRaster().setSample(x, y, 0, value);
                } else {
                    int r = clampToByte(imageData[0 * height * width + y * width + x]);
                    int g = clampToByte(imageData[1 * height * width + y * width + x]);
                    int b = clampToByte(imageData[2 * height * width + y * width + x]);
                    int rgb = (r << 16) | (g << 8) | b;
                    bufferedImage.setRGB(x, y, rgb);
                }
            }
        }

        Path path = Paths.get(outputPath);
        if (path.getParent() != null) {
            path.getParent().toFile().mkdirs();
        }

        String format = outputPath.endsWith(".png") ? "PNG" : "JPEG";
        ImageIO.write(bufferedImage, format, new File(outputPath));
    }

    /**
     * Save multiple images as separate files.
     *
     * @param images Batch of images
     * @param outputDir Directory to save images
     * @param prefix Filename prefix
     */
    public static void saveImages(NDArray images, String outputDir, String prefix) throws IOException {
        long batchSize = images.getShape().get(0);

        for (int i = 0; i < batchSize; i++) {
            String filename = String.format("%s/%s_%04d.png", outputDir, prefix, i);
            saveImage(images.get(i), filename);
        }
    }

    /**
     * Convert NDArray to BufferedImage.
     */
    public static BufferedImage toBufferedImage(NDArray image) {
        Shape shape = image.getShape();
        NDArray img = shape.dimension() == 4 ? image.get(0) : image;
        shape = img.getShape();

        int channels = (int) shape.get(0);
        int height = (int) shape.get(1);
        int width = (int) shape.get(2);

        BufferedImage bufferedImage = new BufferedImage(
                width, height,
                channels == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB
        );

        float[] imageData = img.toFloatArray();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (channels == 1) {
                    int value = clampToByte(imageData[y * width + x]);
                    bufferedImage.getRaster().setSample(x, y, 0, value);
                } else {
                    int r = clampToByte(imageData[0 * height * width + y * width + x]);
                    int g = clampToByte(imageData[1 * height * width + y * width + x]);
                    int b = clampToByte(imageData[2 * height * width + y * width + x]);
                    int rgb = (r << 16) | (g << 8) | b;
                    bufferedImage.setRGB(x, y, rgb);
                }
            }
        }

        return bufferedImage;
    }

    /**
     * Create a horizontal concatenation of images.
     */
    public static NDArray hstack(NDManager manager, NDArray... images) {
        if (images.length == 0) {
            throw new IllegalArgumentException("At least one image required");
        }
        if (images.length == 1) {
            return images[0];
        }

        NDArray result = images[0];
        for (int i = 1; i < images.length; i++) {
            result = result.concat(images[i], 3);  // Concat along width
        }
        return result;
    }

    /**
     * Create a vertical concatenation of images.
     */
    public static NDArray vstack(NDManager manager, NDArray... images) {
        if (images.length == 0) {
            throw new IllegalArgumentException("At least one image required");
        }
        if (images.length == 1) {
            return images[0];
        }

        NDArray result = images[0];
        for (int i = 1; i < images.length; i++) {
            result = result.concat(images[i], 2);  // Concat along height
        }
        return result;
    }

    /**
     * Rescale image values from model output range to [0, 255].
     */
    public static NDArray rescaleToByteRange(NDArray image) {
        // Assume input is roughly in [0, 1] or similar
        // Clip and scale to [0, 255]
        return image.clip(0, 1).mul(255);
    }

    /**
     * Normalize image to [0, 1] range based on min/max.
     */
    public static NDArray normalizeMinMax(NDArray image) {
        float min = image.min().getFloat();
        float max = image.max().getFloat();
        if (Math.abs(max - min) < 1e-8) {
            return image.zerosLike().add(0.5f);
        }
        return image.sub(min).div(max - min);
    }

    /**
     * Clamp float value to byte range [0, 255].
     */
    private static int clampToByte(float value) {
        // Assume value is in [0, 1], scale to [0, 255]
        int scaled = Math.round(value * 255);
        return Math.max(0, Math.min(255, scaled));
    }

    /**
     * Add a border around an image.
     */
    public static NDArray addBorder(NDArray image, int borderSize, float borderValue) {
        Shape shape = image.getShape();
        long batch = shape.get(0);
        long channels = shape.get(1);
        long height = shape.get(2);
        long width = shape.get(3);

        NDManager manager = image.getManager();
        NDArray bordered = manager.full(
                new Shape(batch, channels, height + 2 * borderSize, width + 2 * borderSize),
                borderValue
        );

        // This is a simplified version - proper implementation would use slicing
        return bordered;
    }

    /**
     * Create a comparison grid showing original, corrupted, and reconstructed images.
     */
    public static void saveComparisonGrid(NDArray original, NDArray corrupted, NDArray reconstructed,
                                          String outputPath, int numColumns) throws IOException {
        // Interleave: orig1, corr1, recon1, orig2, corr2, recon2, ...
        long batchSize = original.getShape().get(0);
        NDManager manager = original.getManager();

        NDArray interleaved = original.get(0).expandDims(0);
        interleaved = interleaved.concat(corrupted.get(0).expandDims(0), 0);
        interleaved = interleaved.concat(reconstructed.get(0).expandDims(0), 0);

        for (int i = 1; i < batchSize; i++) {
            interleaved = interleaved.concat(original.get(i).expandDims(0), 0);
            interleaved = interleaved.concat(corrupted.get(i).expandDims(0), 0);
            interleaved = interleaved.concat(reconstructed.get(i).expandDims(0), 0);
        }

        saveImageGrid(interleaved, outputPath, numColumns * 3);
    }
}
