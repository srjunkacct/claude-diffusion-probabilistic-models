package com.technodrome.diffusion.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;

import com.technodrome.diffusion.data.DataPreprocessor;
import com.technodrome.diffusion.data.MnistDataLoader;
import com.technodrome.diffusion.model.DiffusionModel;
import com.technodrome.diffusion.sampling.Sampler;
import com.technodrome.diffusion.util.ImageUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;

/**
 * Training loop for the diffusion model.
 *
 * Implements:
 * - RMSprop optimization with learning rate decay
 * - Gradient clipping
 * - Periodic checkpointing
 * - Sample generation during training
 * - Progress logging
 */
public class Trainer {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    private final DiffusionModel model;
    private final TrainingConfig config;
    private final NDManager manager;
    private final Device device;

    private MnistDataLoader trainLoader;
    private Sampler sampler;
    private ParameterStore parameterStore;

    // RMSprop state
    private java.util.Map<String, NDArray> squaredGradAvg;

    // Training state
    private int currentEpoch = 0;
    private int globalStep = 0;
    private double runningLoss = 0.0;
    private int lossCount = 0;

    // Adaptive learning rate state
    private double currentLearningRate;
    private double bestLoss = Double.MAX_VALUE;
    private int epochsWithoutImprovement = 0;

    /**
     * Create a trainer.
     *
     * @param model Diffusion model to train
     * @param config Training configuration
     * @param manager NDManager for memory management
     */
    public Trainer(DiffusionModel model, TrainingConfig config, NDManager manager) {
        this.model = model;
        this.config = config;
        this.manager = manager;
        this.device = Device.gpu();  // Use GPU if available
        this.squaredGradAvg = new java.util.HashMap<>();
        this.parameterStore = new ParameterStore(manager, false);
        this.currentLearningRate = config.getLearningRate();
    }

    /**
     * Initialize training.
     */
    public void initialize() throws IOException {
        // Create directories
        Files.createDirectories(Paths.get(config.getCheckpointDir()));
        Files.createDirectories(Paths.get(config.getSampleDir()));
        Files.createDirectories(Paths.get(config.getLogDir()));

        // Initialize data loader
        trainLoader = new MnistDataLoader(config.getBatchSize(), true);

        // Initialize model parameters
        Shape inputShape = new Shape(config.getBatchSize(), config.getImageChannels(),
                config.getImageHeight(), config.getImageWidth());
        model.initialize(manager, DataType.FLOAT32, inputShape);
        model.initializeDiffusionParams(manager);

        // Initialize sampler
        sampler = new Sampler(model, manager);

        logger.info("Training initialized with config:\n{}", config);
    }

    /**
     * Run the full training loop.
     */
    public void train() throws Exception {
        logger.info("Starting training for {} epochs", config.getNumEpochs());
        if (config.isUseAdaptiveLR()) {
            logger.info("Using adaptive learning rate with patience={}, reduction factor={}",
                    config.getLrPatience(), config.getLrReductionFactor());
        }

        for (int epoch = 0; epoch < config.getNumEpochs(); epoch++) {
            currentEpoch = epoch;
            double epochLoss = trainEpoch();

            // Adaptive learning rate adjustment
            if (config.isUseAdaptiveLR()) {
                adjustLearningRate(epochLoss);
            } else {
                // Fixed schedule decay
                currentLearningRate = config.getLearningRateAtEpoch(epoch);
                if (epoch > 0 && epoch % config.getLearningRateDecayEpochs() == 0) {
                    logger.info("Learning rate decayed to {}", currentLearningRate);
                }
            }

            // Checkpointing
            if ((epoch + 1) % config.getCheckpointEveryNEpochs() == 0) {
                saveCheckpoint(epoch);
            }

            // Sample generation
            if ((epoch + 1) % config.getSampleEveryNEpochs() == 0) {
                generateSamples(epoch);
            }
        }

        logger.info("Training completed!");
    }

    /**
     * Adjust learning rate based on loss improvement (reduce on plateau).
     */
    private void adjustLearningRate(double epochLoss) {
        // Check if we have meaningful improvement
        double relativeImprovement = (bestLoss - epochLoss) / Math.abs(bestLoss);

        if (epochLoss < bestLoss && relativeImprovement > config.getLrThreshold()) {
            // Loss improved significantly
            bestLoss = epochLoss;
            epochsWithoutImprovement = 0;
            logger.debug("New best loss: {}", String.format("%.4f", bestLoss));
        } else {
            // No significant improvement
            epochsWithoutImprovement++;
            logger.debug("Epochs without improvement: {}/{}", epochsWithoutImprovement, config.getLrPatience());

            if (epochsWithoutImprovement >= config.getLrPatience()) {
                // Reduce learning rate
                double oldLr = currentLearningRate;
                currentLearningRate *= config.getLrReductionFactor();

                // Enforce minimum
                if (currentLearningRate < config.getMinLearningRate()) {
                    currentLearningRate = config.getMinLearningRate();
                }

                if (currentLearningRate < oldLr) {
                    logger.info("Reducing learning rate: {} -> {} (no improvement for {} epochs)",
                            String.format("%.2e", oldLr),
                            String.format("%.2e", currentLearningRate),
                            epochsWithoutImprovement);
                }

                // Reset patience counter
                epochsWithoutImprovement = 0;

                // Update best loss to current to prevent immediate re-triggering
                bestLoss = epochLoss;
            }
        }
    }

    /**
     * Train for one epoch.
     *
     * @return Average loss for this epoch
     */
    private double trainEpoch() throws Exception {
        double epochRunningLoss = 0.0;
        int epochLossCount = 0;
        runningLoss = 0.0;
        lossCount = 0;

        for (Batch batch : trainLoader.iterateBatches(manager)) {
            try (NDManager stepManager = manager.newSubManager()) {
                // Get raw images and duplicate into stepManager (stays on GPU)
                NDArray rawImages = batch.getData().singletonOrThrow();
                NDArray imagesCopy = rawImages.duplicate();
                imagesCopy.attach(stepManager);

                // Preprocess using the copy (intermediates will be on stepManager)
                NDArray images = trainLoader.getPreprocessor().preprocess(imagesCopy);

                float loss = trainStep(images, stepManager);

                // Skip NaN losses (bad batches)
                if (!Float.isNaN(loss)) {
                    runningLoss += loss;
                    lossCount++;
                    epochRunningLoss += loss;
                    epochLossCount++;
                }
                globalStep++;

                if (globalStep % config.getLogEveryNSteps() == 0) {
                    double avgLoss = lossCount > 0 ? runningLoss / lossCount : 0;
                    logger.info("{} Epoch {} Step {} - Loss: {} (lr: {})",
                            Instant.now().toString(), currentEpoch, globalStep,
                            String.format("%.4f", avgLoss),
                            String.format("%.2e", currentLearningRate));
                    runningLoss = 0.0;
                    lossCount = 0;
                }
            } finally {
                batch.close();
            }
        }

        double epochLoss = epochLossCount > 0 ? epochRunningLoss / epochLossCount : 0;
        logger.info("Epoch {} completed - Average Loss: {} (lr: {})",
                currentEpoch, String.format("%.4f", epochLoss),
                String.format("%.2e", currentLearningRate));
        logger.info(Instant.now().toString());

        // Force garbage collection to free native memory
        System.gc();

        return epochLoss;
    }

    /**
     * Perform a single training step.
     *
     * @param images Batch of preprocessed images
     * @param stepManager Sub-manager for this step's memory
     * @return Loss value
     */
    private float trainStep(NDArray images, NDManager stepManager) {
        float lossValue;

        try (GradientCollector gc = stepManager.getEngine().newGradientCollector()) {
            // Forward pass
            NDList output = model.forward(parameterStore, new NDList(images), true);
            NDArray loss = output.singletonOrThrow();

            lossValue = loss.getFloat();

            // Skip backward pass if loss is NaN, Inf, or extremely large
            // Normal MNIST training should have losses < 1000 after a few epochs
            // A loss of 10000+ indicates numerical instability
            if (Float.isNaN(lossValue) || Float.isInfinite(lossValue) || lossValue > 10000) {
                logger.warn("Skipping batch due to extreme loss value: {}", lossValue);
                return Float.NaN;  // Signal to skip this batch
            }

            // Backward pass
            gc.backward(loss);
        }

        // Apply gradients with RMSprop using current (possibly adapted) learning rate
        applyRMSprop(currentLearningRate);

        return lossValue;
    }

    /**
     * Apply RMSprop optimization using GPU tensors with proper memory management.
     */
    private void applyRMSprop(double learningRate) {
        double decay = config.getRmspropDecay();
        double epsilon = config.getRmspropEpsilon();
        double clipNorm = config.getGradientClipNorm();

        try (NDManager optimManager = manager.newSubManager()) {
            // First pass: compute global gradient norm for clipping
            double globalNormSq = 0.0;
            java.util.List<NDArray> gradients = new java.util.ArrayList<>();
            java.util.List<String> paramNames = new java.util.ArrayList<>();
            java.util.List<ai.djl.nn.Parameter> params = new java.util.ArrayList<>();

            for (var pair : model.getParameters()) {
                String name = pair.getKey();
                var param = pair.getValue();

                if (param.requiresGradient()) {
                    NDArray gradientOrig = param.getArray().getGradient();
                    if (gradientOrig == null) {
                        continue;
                    }

                    // Duplicate gradient into optimManager for safe operations
                    NDArray gradient = gradientOrig.duplicate();
                    gradient.attach(optimManager);

                    // Skip if gradient contains NaN or Inf
                    float gradMax = gradient.abs().max().getFloat();
                    if (Float.isNaN(gradMax) || Float.isInfinite(gradMax)) {
                        continue;
                    }

                    float gradNorm = gradient.norm().getFloat();
                    globalNormSq += gradNorm * gradNorm;

                    gradients.add(gradient);
                    paramNames.add(name);
                    params.add(param);
                }
            }

            // Compute global clipping factor
            double globalNorm = Math.sqrt(globalNormSq);
            double clipFactor = 1.0;
            if (clipNorm > 0 && globalNorm > clipNorm) {
                clipFactor = clipNorm / globalNorm;
            }

            // Second pass: apply clipped gradients
            for (int i = 0; i < gradients.size(); i++) {
                NDArray gradient = gradients.get(i);
                String name = paramNames.get(i);
                var param = params.get(i);

                // Apply global clipping factor
                if (clipFactor < 1.0) {
                    gradient.muli(clipFactor);
                }

                // Compute grad^2 in optimManager
                NDArray gradSquared = gradient.square();

                // RMSprop update
                NDArray sqGradAvg = squaredGradAvg.get(name);

                if (sqGradAvg == null) {
                    // First time: initialize squared gradient average on main manager
                    sqGradAvg = gradSquared.mul(1 - decay);
                    sqGradAvg.attach(manager);
                    squaredGradAvg.put(name, sqGradAvg);
                } else {
                    // Update in-place: sqGradAvg = decay * sqGradAvg + (1-decay) * grad^2
                    sqGradAvg.muli(decay).addi(gradSquared.mul(1 - decay));
                }

                // Compute sqrt in optimManager to avoid memory leak
                NDArray sqrtAvg = sqGradAvg.sqrt();
                sqrtAvg.attach(optimManager);

                // Compute update: lr * gradient / (sqrt(sqGradAvg) + epsilon)
                NDArray update = gradient.div(sqrtAvg.add(epsilon)).mul(learningRate);

                // Apply update in-place to parameter
                param.getArray().subi(update);
            }
        }
    }

    /**
     * Save a training checkpoint.
     */
    private void saveCheckpoint(int epoch) {
        try {
            Path checkpointPath = Paths.get(config.getCheckpointDir(), "epoch_" + epoch);
            Files.createDirectories(checkpointPath);

            // Save model parameters
            try (Model djlModel = Model.newInstance("diffusion")) {
                djlModel.setBlock(model);
                djlModel.save(checkpointPath, "diffusion_model");
            }

            logger.info("Checkpoint saved at epoch {}", epoch);
        } catch (Exception e) {
            logger.error("Failed to save checkpoint at epoch {}", epoch, e);
        }
    }

    /**
     * Load a checkpoint.
     */
    public void loadCheckpoint(String path) throws Exception {
        Path checkpointPath = Paths.get(path);
        try (Model djlModel = Model.newInstance("diffusion")) {
            djlModel.setBlock(model);
            djlModel.load(checkpointPath, "diffusion_model");
        }
        logger.info("Checkpoint loaded from {}", path);
    }

    /**
     * Generate and save samples.
     */
    private void generateSamples(int epoch) {
        try {
            logger.info("Generating samples at epoch {}", epoch);

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
                logger.error("Model has {} NaN and {} Inf parameters - skipping sample generation", nanParams, infParams);
                return;
            }

            NDArray samples = sampler.generateSamples(
                    config.getNumSamplesToGenerate(),
                    parameterStore);

            // Log raw model output statistics
            logger.info("Raw model output - min: {}, max: {}, mean: {}, std: {}",
                    String.format("%.4f", samples.min().getFloat()),
                    String.format("%.4f", samples.max().getFloat()),
                    String.format("%.4f", samples.mean().getFloat()),
                    String.format("%.4f", samples.sub(samples.mean()).pow(2).mean().sqrt().getFloat()));

            // Convert from model output space back to image space
            samples = trainLoader.getPreprocessor().inversePreprocess(samples);

            // Log post-processing statistics
            logger.info("After inverse preprocess - min: {}, max: {}, mean: {}",
                    String.format("%.4f", samples.min().getFloat()),
                    String.format("%.4f", samples.max().getFloat()),
                    String.format("%.4f", samples.mean().getFloat()));

            // Save as image grid
            Path samplePath = Paths.get(config.getSampleDir(),
                    String.format("samples_epoch_%04d.png", epoch));
            ImageUtils.saveImageGrid(samples, samplePath.toString(), 8);

            // Also save a min-max normalized version for visualization
            NDArray normalizedSamples = DataPreprocessor.normalizeToUnitRange(samples);
            Path normalizedPath = Paths.get(config.getSampleDir(),
                    String.format("samples_epoch_%04d_normalized.png", epoch));
            ImageUtils.saveImageGrid(normalizedSamples, normalizedPath.toString(), 8);

            logger.info("Samples saved to {}", samplePath);
        } catch (Exception e) {
            logger.error("Failed to generate samples at epoch {}", epoch, e);
        }
    }

    /**
     * Get current training state for resuming.
     */
    public TrainingState getState() {
        return new TrainingState(currentEpoch, globalStep, runningLoss, lossCount,
                currentLearningRate, bestLoss, epochsWithoutImprovement);
    }

    /**
     * Set training state for resuming.
     */
    public void setState(TrainingState state) {
        this.currentEpoch = state.epoch;
        this.globalStep = state.globalStep;
        this.runningLoss = state.runningLoss;
        this.lossCount = state.lossCount;
        this.currentLearningRate = state.currentLearningRate;
        this.bestLoss = state.bestLoss;
        this.epochsWithoutImprovement = state.epochsWithoutImprovement;
    }

    /**
     * Training state for checkpointing.
     */
    public static class TrainingState {
        public final int epoch;
        public final int globalStep;
        public final double runningLoss;
        public final int lossCount;
        public final double currentLearningRate;
        public final double bestLoss;
        public final int epochsWithoutImprovement;

        public TrainingState(int epoch, int globalStep, double runningLoss, int lossCount,
                             double currentLearningRate, double bestLoss, int epochsWithoutImprovement) {
            this.epoch = epoch;
            this.globalStep = globalStep;
            this.runningLoss = runningLoss;
            this.lossCount = lossCount;
            this.currentLearningRate = currentLearningRate;
            this.bestLoss = bestLoss;
            this.epochsWithoutImprovement = epochsWithoutImprovement;
        }
    }

    // Getters
    public DiffusionModel getModel() {
        return model;
    }

    public TrainingConfig getConfig() {
        return config;
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public int getGlobalStep() {
        return globalStep;
    }

    public Sampler getSampler() {
        return sampler;
    }

    public ParameterStore getParameterStore() {
        return parameterStore;
    }

    public double getCurrentLearningRate() {
        return currentLearningRate;
    }
}
