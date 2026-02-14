package com.technodrome.diffusion;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;

import com.technodrome.diffusion.data.DataPreprocessor;
import com.technodrome.diffusion.data.MnistDataLoader;
import com.technodrome.diffusion.model.DiffusionModel;
import com.technodrome.diffusion.sampling.InpaintMask;
import com.technodrome.diffusion.sampling.Sampler;
import com.technodrome.diffusion.training.Trainer;
import com.technodrome.diffusion.training.TrainingConfig;
import com.technodrome.diffusion.util.ImageUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Paths;

/**
 * Main entry point for the Diffusion Probabilistic Models application.
 *
 * Supports:
 * - Training mode: Train a new model from scratch
 * - Sample mode: Generate new images using a trained model
 * - Inpaint mode: Fill in masked regions of images
 * - Denoise mode: Reconstruct images from noisy versions
 */
public class DiffusionApp {

    private static final Logger logger = LoggerFactory.getLogger(DiffusionApp.class);

    public static void main(String[] args) {
        try {
            if (args.length == 0) {
                printUsage();
                return;
            }

            String mode = args[0].toLowerCase();

            switch (mode) {
                case "train":
                    runTraining(args);
                    break;
                case "sample":
                    runSampling(args);
                    break;
                case "inpaint":
                    runInpainting(args);
                    break;
                case "denoise":
                    runDenoising(args);
                    break;
                case "test":
                    runTest(args);
                    break;
                default:
                    logger.error("Unknown mode: {}", mode);
                    printUsage();
            }
        } catch (Exception e) {
            logger.error("Error running application", e);
            System.exit(1);
        }
    }

    private static void printUsage() {
        System.out.println("Diffusion Probabilistic Models - Java/DJL Implementation");
        System.out.println();
        System.out.println("Usage: java -jar diffusion.jar <mode> [options]");
        System.out.println();
        System.out.println("Modes:");
        System.out.println("  train              Train a new model");
        System.out.println("    --epochs N       Number of epochs (default: 2000)");
        System.out.println("    --batch-size N   Batch size (default: 512)");
        System.out.println("    --lr N           Learning rate (default: 0.001)");
        System.out.println("    --checkpoint DIR Checkpoint directory (default: checkpoints)");
        System.out.println();
        System.out.println("  sample             Generate samples from trained model");
        System.out.println("    --model PATH     Path to model checkpoint");
        System.out.println("    --num N          Number of samples to generate (default: 64)");
        System.out.println("    --output PATH    Output image path (default: samples.png)");
        System.out.println();
        System.out.println("  inpaint            Inpaint masked regions");
        System.out.println("    --model PATH     Path to model checkpoint");
        System.out.println("    --mask TYPE      Mask type: center, half, random (default: center)");
        System.out.println("    --output PATH    Output image path");
        System.out.println();
        System.out.println("  denoise            Denoise corrupted images");
        System.out.println("    --model PATH     Path to model checkpoint");
        System.out.println("    --noise-level N  Corruption level 0-1000 (default: 500)");
        System.out.println("    --output PATH    Output image path");
        System.out.println();
        System.out.println("  test               Run a quick test with small model");
        System.out.println();
    }

    private static void runTraining(String[] args) throws Exception {
        logger.info("Starting training mode");

        // Parse arguments
        TrainingConfig config = TrainingConfig.createMnistDefault();

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--epochs":
                    config.setNumEpochs(Integer.parseInt(args[++i]));
                    break;
                case "--batch-size":
                    config.setBatchSize(Integer.parseInt(args[++i]));
                    break;
                case "--lr":
                    config.setLearningRate(Double.parseDouble(args[++i]));
                    break;
                case "--checkpoint":
                    config.setCheckpointDir(args[++i]);
                    break;
                case "--small":
                    config = TrainingConfig.createSmallTest();
                    break;
            }
        }

        logger.info("Training configuration:\n{}", config);

        try (NDManager manager = NDManager.newBaseManager()) {
            // Create model
            DiffusionModel model = DiffusionModel.createMnistModel(
                    config.getTrajectoryLength(),
                    config.getNumTemporalBasis());

            // Create trainer
            Trainer trainer = new Trainer(model, config, manager);
            trainer.initialize();

            // Run training
            trainer.train();

            logger.info("Training completed successfully!");
        }
    }

    private static void runSampling(String[] args) throws Exception {
        logger.info("Starting sampling mode");

        String modelPath = null;
        int numSamples = 64;
        String outputPath = "samples.png";

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--model":
                    modelPath = args[++i];
                    break;
                case "--num":
                    numSamples = Integer.parseInt(args[++i]);
                    break;
                case "--output":
                    outputPath = args[++i];
                    break;
            }
        }

        try (NDManager manager = NDManager.newBaseManager();
             Model djlModel = Model.newInstance("diffusion")) {
            // Create model
            TrainingConfig config = TrainingConfig.createMnistDefault();
            DiffusionModel model = DiffusionModel.createMnistModel(
                    config.getTrajectoryLength(),
                    config.getNumTemporalBasis());

            // Initialize model
            Shape inputShape = new Shape(numSamples, 1, 28, 28);
            model.initialize(manager, DataType.FLOAT32, inputShape);
            model.initializeDiffusionParams(manager);

            // Load checkpoint if provided
            if (modelPath != null) {
                djlModel.setBlock(model);
                djlModel.load(Paths.get(modelPath), "diffusion_model");
                logger.info("Loaded model from {}", modelPath);
            } else {
                logger.warn("No model checkpoint provided, using random initialization");
            }

            // Create sampler
            Sampler sampler = new Sampler(model, manager);
            ParameterStore parameterStore = new ParameterStore(manager, false);

            // Check for NaN in model weights before sampling
            int nanParams = 0;
            int infParams = 0;
            for (var pair : model.getParameters()) {
                NDArray param = pair.getValue().getArray();
                float maxVal = param.abs().max().getFloat();
                if (Float.isNaN(maxVal)) {
                    nanParams++;
                    logger.warn("NaN detected in parameter: {}", pair.getKey());
                } else if (Float.isInfinite(maxVal)) {
                    infParams++;
                    logger.warn("Inf detected in parameter: {}", pair.getKey());
                }
            }
            if (nanParams > 0 || infParams > 0) {
                logger.error("Model has {} NaN and {} Inf parameters - cannot generate samples", nanParams, infParams);
                return;
            }
            logger.info("Model weights check passed - no NaN/Inf detected");

            // Generate samples
            logger.info("Generating {} samples...", numSamples);
            NDArray samples = sampler.generateSamples(numSamples, parameterStore);

            // Log raw model output statistics
            logger.info("Raw model output - min: {}, max: {}, mean: {}, std: {}",
                    String.format("%.4f", samples.min().getFloat()),
                    String.format("%.4f", samples.max().getFloat()),
                    String.format("%.4f", samples.mean().getFloat()),
                    String.format("%.4f", samples.sub(samples.mean()).pow(2).mean().sqrt().getFloat()));

            // Convert to image range
            DataPreprocessor preprocessor = DataPreprocessor.createMnistPreprocessor();
            samples = preprocessor.inversePreprocess(samples);

            // Log post-processing statistics
            logger.info("After inverse preprocess - min: {}, max: {}, mean: {}",
                    String.format("%.4f", samples.min().getFloat()),
                    String.format("%.4f", samples.max().getFloat()),
                    String.format("%.4f", samples.mean().getFloat()));

            // Save
            int gridCols = (int) Math.ceil(Math.sqrt(numSamples));
            ImageUtils.saveImageGrid(samples, outputPath, gridCols);
            logger.info("Samples saved to {}", outputPath);
        }
    }

    private static void runInpainting(String[] args) throws Exception {
        logger.info("Starting inpainting mode");

        String modelPath = null;
        String maskType = "center";
        String outputPath = "inpainted.png";
        int numSamples = 16;

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--model":
                    modelPath = args[++i];
                    break;
                case "--mask":
                    maskType = args[++i];
                    break;
                case "--output":
                    outputPath = args[++i];
                    break;
                case "--num":
                    numSamples = Integer.parseInt(args[++i]);
                    break;
            }
        }

        try (NDManager manager = NDManager.newBaseManager();
             Model djlModel = Model.newInstance("diffusion")) {
            // Setup model
            TrainingConfig config = TrainingConfig.createMnistDefault();
            DiffusionModel model = DiffusionModel.createMnistModel(
                    config.getTrajectoryLength(),
                    config.getNumTemporalBasis());

            Shape inputShape = new Shape(numSamples, 1, 28, 28);
            model.initialize(manager, DataType.FLOAT32, inputShape);
            model.initializeDiffusionParams(manager);

            if (modelPath != null) {
                djlModel.setBlock(model);
                djlModel.load(Paths.get(modelPath), "diffusion_model");
                logger.info("Loaded model from {}", modelPath);
            }

            // Load some test images
            MnistDataLoader loader = new MnistDataLoader(numSamples, false);
            loader.prepare(manager);
            NDArray originals = loader.getBatch(manager);

            // Create mask
            InpaintMask maskGen = InpaintMask.createMnist();
            NDArray mask;
            switch (maskType) {
                case "center":
                    mask = maskGen.centerMask(manager, numSamples, 5);
                    break;
                case "half":
                    mask = maskGen.halfMask(manager, numSamples, true);
                    break;
                case "random":
                    mask = maskGen.randomMask(manager, numSamples, 0.5f);
                    break;
                default:
                    mask = maskGen.centerMask(manager, numSamples, 5);
            }

            // Run inpainting
            Sampler sampler = new Sampler(model, manager);
            ParameterStore parameterStore = new ParameterStore(manager, false);

            NDArray inpainted = sampler.inpaint(originals, mask, parameterStore);

            // Save comparison
            DataPreprocessor preprocessor = DataPreprocessor.createMnistPreprocessor();
            originals = preprocessor.inversePreprocess(originals);
            inpainted = preprocessor.inversePreprocess(inpainted);

            ImageUtils.saveComparisonGrid(originals, originals.mul(mask), inpainted,
                    outputPath, 4);
            logger.info("Inpainting results saved to {}", outputPath);
        }
    }

    private static void runDenoising(String[] args) throws Exception {
        logger.info("Starting denoising mode");

        String modelPath = null;
        int noiseLevel = 500;
        String outputPath = "denoised.png";
        int numSamples = 16;

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--model":
                    modelPath = args[++i];
                    break;
                case "--noise-level":
                    noiseLevel = Integer.parseInt(args[++i]);
                    break;
                case "--output":
                    outputPath = args[++i];
                    break;
                case "--num":
                    numSamples = Integer.parseInt(args[++i]);
                    break;
            }
        }

        try (NDManager manager = NDManager.newBaseManager();
             Model djlModel = Model.newInstance("diffusion")) {
            // Setup model
            TrainingConfig config = TrainingConfig.createMnistDefault();
            DiffusionModel model = DiffusionModel.createMnistModel(
                    config.getTrajectoryLength(),
                    config.getNumTemporalBasis());

            Shape inputShape = new Shape(numSamples, 1, 28, 28);
            model.initialize(manager, DataType.FLOAT32, inputShape);
            model.initializeDiffusionParams(manager);

            if (modelPath != null) {
                djlModel.setBlock(model);
                djlModel.load(Paths.get(modelPath), "diffusion_model");
                logger.info("Loaded model from {}", modelPath);
            }

            // Load test images
            MnistDataLoader loader = new MnistDataLoader(numSamples, false);
            loader.prepare(manager);
            NDArray originals = loader.getBatch(manager);

            // Add noise
            NDArray timesteps = manager.full(new Shape(numSamples), noiseLevel);
            NDArray noise = manager.randomNormal(originals.getShape());
            NDArray noisy = model.getForwardDiffusionSample(originals, timesteps.toType(
                    ai.djl.ndarray.types.DataType.INT32, false), noise).get(0);

            // Denoise
            Sampler sampler = new Sampler(model, manager);
            ParameterStore parameterStore = new ParameterStore(manager, false);

            NDArray denoised = sampler.denoise(noisy, noiseLevel, parameterStore);

            // Save comparison
            DataPreprocessor preprocessor = DataPreprocessor.createMnistPreprocessor();
            originals = preprocessor.inversePreprocess(originals);
            noisy = preprocessor.inversePreprocess(noisy);
            denoised = preprocessor.inversePreprocess(denoised);

            ImageUtils.saveComparisonGrid(originals, noisy, denoised, outputPath, 4);
            logger.info("Denoising results saved to {}", outputPath);
        }
    }

    private static void runTest(String[] args) throws Exception {
        logger.info("Running quick test...");

        try (NDManager manager = NDManager.newBaseManager()) {
            // Create small test configuration
            TrainingConfig config = TrainingConfig.createSmallTest();
            config.setNumEpochs(2);
            config.setBatchSize(32);

            logger.info("Test configuration:\n{}", config);

            // Create model
            DiffusionModel model = DiffusionModel.createMnistModel(
                    config.getTrajectoryLength(),
                    config.getNumTemporalBasis());

            // Initialize
            Shape inputShape = new Shape(config.getBatchSize(), 1, 28, 28);
            model.initialize(manager, DataType.FLOAT32, inputShape);
            model.initializeDiffusionParams(manager);

            logger.info("Model initialized successfully");

            // Test forward pass
            NDArray testInput = manager.randomNormal(inputShape);
            ParameterStore parameterStore = new ParameterStore(manager, false);

            logger.info("Testing forward pass...");
            NDArray loss = model.getNegLBound(testInput, parameterStore, false);
            logger.info("Initial loss: {}", loss.getFloat());

            // Test sampling (just a few steps)
            logger.info("Testing sampling (10 steps)...");
            Sampler sampler = new Sampler(model, manager);
            NDArray samples = manager.randomNormal(new Shape(4, 1, 28, 28));

            for (int t = config.getTrajectoryLength() - 1; t >= config.getTrajectoryLength() - 10; t--) {
                int[] timesteps = new int[4];
                java.util.Arrays.fill(timesteps, t);
                NDArray tArray = manager.create(timesteps);

                var muSigma = model.getMuSigma(samples, tArray, parameterStore, false);
                NDArray mu = muSigma.get(0);
                NDArray sigma = muSigma.get(1);

                NDArray noise = manager.randomNormal(samples.getShape());
                samples = mu.add(sigma.mul(noise));
            }

            logger.info("Test completed successfully!");
            logger.info("Sample shape: {}", samples.getShape());
            logger.info("Sample mean: {}, std: {}",
                    samples.mean().getFloat(),
                    samples.sub(samples.mean()).pow(2).mean().sqrt().getFloat());
        }
    }
}
