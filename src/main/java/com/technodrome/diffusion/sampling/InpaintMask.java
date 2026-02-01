package com.technodrome.diffusion.sampling;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Inpainting mask generation for conditional image generation.
 *
 * Creates masks that specify which regions of an image should be:
 * - Generated (mask = 0): Sampled freely by the model
 * - Preserved (mask = 1): Kept from the original image
 *
 * Supports various mask patterns useful for image inpainting tasks.
 */
public class InpaintMask {

    private final int height;
    private final int width;

    /**
     * Create an inpaint mask generator.
     *
     * @param height Image height
     * @param width Image width
     */
    public InpaintMask(int height, int width) {
        this.height = height;
        this.width = width;
    }

    /**
     * Create for MNIST-sized images.
     */
    public static InpaintMask createMnist() {
        return new InpaintMask(28, 28);
    }

    /**
     * Generate a rectangular mask.
     *
     * @param manager NDManager
     * @param batchSize Batch size
     * @param top Top row of rectangle
     * @param left Left column of rectangle
     * @param maskHeight Height of masked region
     * @param maskWidth Width of masked region
     * @param invert If true, preserve inside and generate outside
     * @return Mask array of shape (batchSize, 1, height, width)
     */
    public NDArray rectangleMask(NDManager manager, int batchSize,
                                  int top, int left, int maskHeight, int maskWidth,
                                  boolean invert) {
        float[][] mask = new float[height][width];

        // Fill with 1s (preserve all)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                mask[i][j] = 1.0f;
            }
        }

        // Set rectangle to 0 (generate)
        for (int i = top; i < Math.min(top + maskHeight, height); i++) {
            for (int j = left; j < Math.min(left + maskWidth, width); j++) {
                mask[i][j] = 0.0f;
            }
        }

        if (invert) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    mask[i][j] = 1.0f - mask[i][j];
                }
            }
        }

        // Create NDArray and expand to batch
        NDArray maskArray = create2DMask(manager, mask);
        return maskArray.tile(new long[]{batchSize, 1, 1, 1});
    }

    /**
     * Generate a half-image mask (left or right half).
     *
     * @param manager NDManager
     * @param batchSize Batch size
     * @param preserveLeft If true, preserve left half; otherwise preserve right half
     * @return Mask array
     */
    public NDArray halfMask(NDManager manager, int batchSize, boolean preserveLeft) {
        float[][] mask = new float[height][width];
        int midpoint = width / 2;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (preserveLeft) {
                    mask[i][j] = j < midpoint ? 1.0f : 0.0f;
                } else {
                    mask[i][j] = j >= midpoint ? 1.0f : 0.0f;
                }
            }
        }

        NDArray maskArray = create2DMask(manager, mask);
        return maskArray.tile(new long[]{batchSize, 1, 1, 1});
    }

    /**
     * Generate a center mask (preserve border, generate center).
     *
     * @param manager NDManager
     * @param batchSize Batch size
     * @param borderSize Size of border to preserve
     * @return Mask array
     */
    public NDArray centerMask(NDManager manager, int batchSize, int borderSize) {
        float[][] mask = new float[height][width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                boolean inBorder = i < borderSize || i >= height - borderSize ||
                        j < borderSize || j >= width - borderSize;
                mask[i][j] = inBorder ? 1.0f : 0.0f;
            }
        }

        NDArray maskArray = create2DMask(manager, mask);
        return maskArray.tile(new long[]{batchSize, 1, 1, 1});
    }

    /**
     * Generate a random mask with specified density.
     *
     * @param manager NDManager
     * @param batchSize Batch size
     * @param preserveRatio Ratio of pixels to preserve (0 to 1)
     * @return Mask array
     */
    public NDArray randomMask(NDManager manager, int batchSize, float preserveRatio) {
        NDArray random = manager.randomUniform(0, 1,
                new Shape(batchSize, 1, height, width));
        return random.lt(preserveRatio).toType(DataType.FLOAT32, false);
    }

    /**
     * Generate a checkerboard mask.
     *
     * @param manager NDManager
     * @param batchSize Batch size
     * @param cellSize Size of checkerboard cells
     * @return Mask array
     */
    public NDArray checkerboardMask(NDManager manager, int batchSize, int cellSize) {
        float[][] mask = new float[height][width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int cellRow = i / cellSize;
                int cellCol = j / cellSize;
                mask[i][j] = ((cellRow + cellCol) % 2 == 0) ? 1.0f : 0.0f;
            }
        }

        NDArray maskArray = create2DMask(manager, mask);
        return maskArray.tile(new long[]{batchSize, 1, 1, 1});
    }

    /**
     * Generate a circular mask.
     *
     * @param manager NDManager
     * @param batchSize Batch size
     * @param centerY Y coordinate of circle center
     * @param centerX X coordinate of circle center
     * @param radius Radius of circle
     * @param preserveInside If true, preserve inside circle; otherwise preserve outside
     * @return Mask array
     */
    public NDArray circleMask(NDManager manager, int batchSize,
                               float centerY, float centerX, float radius,
                               boolean preserveInside) {
        float[][] mask = new float[height][width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float dy = i - centerY;
                float dx = j - centerX;
                float distance = (float) Math.sqrt(dy * dy + dx * dx);
                boolean inside = distance <= radius;

                if (preserveInside) {
                    mask[i][j] = inside ? 1.0f : 0.0f;
                } else {
                    mask[i][j] = inside ? 0.0f : 1.0f;
                }
            }
        }

        NDArray maskArray = create2DMask(manager, mask);
        return maskArray.tile(new long[]{batchSize, 1, 1, 1});
    }

    /**
     * Generate horizontal stripe mask.
     *
     * @param manager NDManager
     * @param batchSize Batch size
     * @param stripeHeight Height of each stripe
     * @return Mask array
     */
    public NDArray horizontalStripeMask(NDManager manager, int batchSize, int stripeHeight) {
        float[][] mask = new float[height][width];

        for (int i = 0; i < height; i++) {
            float value = ((i / stripeHeight) % 2 == 0) ? 1.0f : 0.0f;
            for (int j = 0; j < width; j++) {
                mask[i][j] = value;
            }
        }

        NDArray maskArray = create2DMask(manager, mask);
        return maskArray.tile(new long[]{batchSize, 1, 1, 1});
    }

    /**
     * Generate a mask that preserves only specific rows (for denoising demo).
     */
    public NDArray preserveRowsMask(NDManager manager, int batchSize, int[] rows) {
        float[][] mask = new float[height][width];

        // Default to generate (0)
        for (int row : rows) {
            if (row >= 0 && row < height) {
                for (int j = 0; j < width; j++) {
                    mask[row][j] = 1.0f;
                }
            }
        }

        NDArray maskArray = create2DMask(manager, mask);
        return maskArray.tile(new long[]{batchSize, 1, 1, 1});
    }

    /**
     * Create a 2D mask as NDArray.
     */
    private NDArray create2DMask(NDManager manager, float[][] mask) {
        float[] flat = new float[height * width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                flat[i * width + j] = mask[i][j];
            }
        }
        return manager.create(flat, new Shape(1, 1, height, width));
    }

    /**
     * Combine two masks (logical AND).
     */
    public static NDArray combineMasks(NDArray mask1, NDArray mask2) {
        return mask1.mul(mask2);
    }

    /**
     * Invert a mask.
     */
    public static NDArray invertMask(NDArray mask) {
        return mask.neg().add(1);
    }

    /**
     * Apply mask to image: masked regions get replaced with replacement values.
     *
     * @param image Original image
     * @param mask Mask (1 = keep, 0 = replace)
     * @param replacement Replacement values
     * @return Masked image
     */
    public static NDArray applyMask(NDArray image, NDArray mask, NDArray replacement) {
        return image.mul(mask).add(replacement.mul(invertMask(mask)));
    }

    // Getters
    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }
}
