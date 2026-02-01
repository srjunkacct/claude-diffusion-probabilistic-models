package com.technodrome.diffusion.sampling;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;

import com.technodrome.diffusion.model.DiffusionModel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sampler for generating images via reverse diffusion.
 *
 * Implements:
 * - Unconditional image generation (from pure noise)
 * - Inpainting (conditional generation with masks)
 * - Denoising (from partially corrupted images)
 */
public class Sampler {

    private static final Logger logger = LoggerFactory.getLogger(Sampler.class);

    private final DiffusionModel model;
    private final NDManager manager;
    private final int trajectoryLength;
    private final int imageChannels;
    private final int imageHeight;
    private final int imageWidth;

    /**
     * Create a sampler.
     *
     * @param model Trained diffusion model
     * @param manager NDManager for memory management
     */
    public Sampler(DiffusionModel model, NDManager manager) {
        this.model = model;
        this.manager = manager;
        this.trajectoryLength = model.getTrajectoryLength();
        this.imageChannels = 1;  // MNIST
        this.imageHeight = 28;
        this.imageWidth = 28;
    }

    /**
     * Generate samples via unconditional reverse diffusion.
     *
     * Starting from pure Gaussian noise, iteratively applies the learned
     * reverse process to generate clean images.
     *
     * @param numSamples Number of samples to generate
     * @param parameterStore Parameter store with model weights
     * @return Generated images of shape (numSamples, channels, height, width)
     */
    public NDArray generateSamples(int numSamples, ParameterStore parameterStore) {
        logger.info("Generating {} samples...", numSamples);

        // Start from pure Gaussian noise
        Shape sampleShape = new Shape(numSamples, imageChannels, imageHeight, imageWidth);
        NDArray x = manager.randomNormal(sampleShape);

        // Run reverse diffusion from T to 0
        for (int t = trajectoryLength - 1; t >= 0; t--) {
            x = diffusionStep(x, t, parameterStore, null, null);

            if (t % 100 == 0) {
                logger.debug("Sampling step {}/{}", trajectoryLength - t, trajectoryLength);
            }
        }

        logger.info("Sample generation complete");
        return x;
    }

    /**
     * Generate samples with progress callback.
     */
    public NDArray generateSamples(int numSamples, ParameterStore parameterStore,
                                   SamplingCallback callback) {
        Shape sampleShape = new Shape(numSamples, imageChannels, imageHeight, imageWidth);
        NDArray x = manager.randomNormal(sampleShape);

        for (int t = trajectoryLength - 1; t >= 0; t--) {
            x = diffusionStep(x, t, parameterStore, null, null);

            if (callback != null) {
                callback.onStep(t, trajectoryLength, x);
            }
        }

        return x;
    }

    /**
     * Inpainting: generate content for masked regions while preserving unmasked regions.
     *
     * @param originalImage Original image with regions to preserve
     * @param mask Binary mask (1 = preserve, 0 = generate)
     * @param parameterStore Parameter store
     * @return Inpainted image
     */
    public NDArray inpaint(NDArray originalImage, NDArray mask,
                           ParameterStore parameterStore) {
        logger.info("Starting inpainting...");

        long batchSize = originalImage.getShape().get(0);
        Shape sampleShape = originalImage.getShape();

        // Start from noise in masked regions, original in unmasked
        NDArray noise = manager.randomNormal(sampleShape);
        NDArray x = InpaintMask.applyMask(originalImage, mask, noise);

        // Run reverse diffusion with masking at each step
        for (int t = trajectoryLength - 1; t >= 0; t--) {
            x = diffusionStep(x, t, parameterStore, originalImage, mask);
        }

        logger.info("Inpainting complete");
        return x;
    }

    /**
     * Denoise an image corrupted at a specific timestep.
     *
     * @param noisyImage Corrupted image at timestep t
     * @param startTimestep Timestep at which corruption occurred
     * @param parameterStore Parameter store
     * @return Denoised image
     */
    public NDArray denoise(NDArray noisyImage, int startTimestep,
                           ParameterStore parameterStore) {
        logger.info("Denoising from timestep {}...", startTimestep);

        NDArray x = noisyImage;

        // Run reverse diffusion from startTimestep to 0
        for (int t = startTimestep; t >= 0; t--) {
            x = diffusionStep(x, t, parameterStore, null, null);
        }

        logger.info("Denoising complete");
        return x;
    }

    /**
     * Perform a single reverse diffusion step.
     *
     * p(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_theta(x_t, t))
     *
     * @param xt Current noisy sample at time t
     * @param t Current timestep
     * @param parameterStore Parameter store
     * @param originalImage Optional original image for inpainting
     * @param mask Optional mask for inpainting
     * @return Sample at time t-1
     */
    private NDArray diffusionStep(NDArray xt, int t, ParameterStore parameterStore,
                                   NDArray originalImage, NDArray mask) {
        long batchSize = xt.getShape().get(0);

        // Create timestep array
        int[] timesteps = new int[(int) batchSize];
        java.util.Arrays.fill(timesteps, t);
        NDArray tArray = manager.create(timesteps);

        // Get mu and sigma from model
        NDList muSigma = model.getMuSigma(xt, tArray, parameterStore, false);
        NDArray mu = muSigma.get(0);
        NDArray sigma = muSigma.get(1);

        // Sample x_{t-1} from N(mu, sigma^2)
        NDArray noise = manager.randomNormal(xt.getShape());
        NDArray xPrev;

        if (t > 0) {
            // Add noise for t > 0
            xPrev = mu.add(sigma.mul(noise));
        } else {
            // No noise at t = 0 (final step)
            xPrev = mu;
        }

        // Apply mask for inpainting if provided
        if (mask != null && originalImage != null) {
            // Regions with mask=1 should be from original (appropriately noised)
            // Regions with mask=0 should be from sampling

            if (t > 0) {
                // Get the noised version of original at current timestep
                NDList forwardResult = model.getForwardDiffusionSample(originalImage, tArray, null);
                NDArray originalNoised = forwardResult.get(0);

                // Blend: mask=1 keeps original noised, mask=0 keeps sampled
                xPrev = InpaintMask.applyMask(originalNoised, mask, xPrev);
            } else {
                // At t=0, use clean original for masked regions
                xPrev = InpaintMask.applyMask(originalImage, mask, xPrev);
            }
        }

        return xPrev;
    }

    /**
     * Generate samples and intermediate states for visualization.
     *
     * @param numSamples Number of samples
     * @param numIntermediates Number of intermediate states to save
     * @param parameterStore Parameter store
     * @return Array of NDArrays at different timesteps
     */
    public NDArray[] generateWithIntermediates(int numSamples, int numIntermediates,
                                                ParameterStore parameterStore) {
        int stepInterval = trajectoryLength / numIntermediates;
        NDArray[] intermediates = new NDArray[numIntermediates + 1];
        int intermediateIdx = 0;

        Shape sampleShape = new Shape(numSamples, imageChannels, imageHeight, imageWidth);
        NDArray x = manager.randomNormal(sampleShape);
        intermediates[intermediateIdx++] = x.duplicate();

        for (int t = trajectoryLength - 1; t >= 0; t--) {
            x = diffusionStep(x, t, parameterStore, null, null);

            if ((trajectoryLength - 1 - t) % stepInterval == 0 && intermediateIdx < intermediates.length) {
                intermediates[intermediateIdx++] = x.duplicate();
            }
        }

        // Ensure final sample is included
        if (intermediateIdx < intermediates.length) {
            intermediates[intermediateIdx] = x.duplicate();
        }

        return intermediates;
    }

    /**
     * Interpolate between two images in latent space.
     *
     * @param image1 First image
     * @param image2 Second image
     * @param numSteps Number of interpolation steps
     * @param noiseLevel Timestep to corrupt images to
     * @param parameterStore Parameter store
     * @return Interpolated images
     */
    public NDArray interpolate(NDArray image1, NDArray image2, int numSteps,
                               int noiseLevel, ParameterStore parameterStore) {
        // Add noise to both images at the same level
        NDArray t = manager.create(new int[]{noiseLevel});
        NDList forward1 = model.getForwardDiffusionSample(image1, t, null);
        NDList forward2 = model.getForwardDiffusionSample(image2, t, null);
        NDArray noisy1 = forward1.get(0);
        NDArray noisy2 = forward2.get(0);

        // Interpolate in noisy space
        NDArray[] interpolated = new NDArray[numSteps];
        for (int i = 0; i < numSteps; i++) {
            float alpha = (float) i / (numSteps - 1);
            NDArray blended = noisy1.mul(1 - alpha).add(noisy2.mul(alpha));
            interpolated[i] = denoise(blended, noiseLevel, parameterStore);
        }

        // Stack along batch dimension
        NDArray result = interpolated[0];
        for (int i = 1; i < numSteps; i++) {
            result = result.concat(interpolated[i], 0);
        }

        return result;
    }

    /**
     * Callback interface for sampling progress.
     */
    public interface SamplingCallback {
        void onStep(int currentStep, int totalSteps, NDArray currentSample);
    }

    // Getters
    public DiffusionModel getModel() {
        return model;
    }

    public int getTrajectoryLength() {
        return trajectoryLength;
    }
}
