package com.technodrome.diffusion.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;

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
        model.prepare(new Shape[]{inputShape});
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

        for (int epoch = 0; epoch < config.getNumEpochs(); epoch++) {
            currentEpoch = epoch;
            trainEpoch();

            // Learning rate decay
            double currentLr = config.getLearningRateAtEpoch(epoch);
            if (epoch > 0 && epoch % config.getLearningRateDecayEpochs() == 0) {
                logger.info("Learning rate decayed to {}", currentLr);
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
     * Train for one epoch.
     */
    private void trainEpoch() throws Exception {
        runningLoss = 0.0;
        lossCount = 0;

        for (Batch batch : trainLoader.iterateBatches(manager)) {
            try {
                NDArray images = trainLoader.preprocessBatch(batch);
                float loss = trainStep(images);

                runningLoss += loss;
                lossCount++;
                globalStep++;

                if (globalStep % config.getLogEveryNSteps() == 0) {
                    double avgLoss = runningLoss / lossCount;
                    logger.info("Epoch {} Step {} - Loss: {:.4f}", currentEpoch, globalStep, avgLoss);
                    runningLoss = 0.0;
                    lossCount = 0;
                }
            } finally {
                batch.close();
            }
        }

        double epochLoss = lossCount > 0 ? runningLoss / lossCount : 0;
        logger.info("Epoch {} completed - Average Loss: {:.4f}", currentEpoch, epochLoss);
    }

    /**
     * Perform a single training step.
     *
     * @param images Batch of preprocessed images
     * @return Loss value
     */
    private float trainStep(NDArray images) {
        float lossValue;

        try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
            // Forward pass
            NDList output = model.forward(parameterStore, new NDList(images), true);
            NDArray loss = output.singletonOrThrow();

            // Backward pass
            gc.backward(loss);
            lossValue = loss.getFloat();
        }

        // Apply gradients with RMSprop
        double lr = config.getLearningRateAtEpoch(currentEpoch);
        applyRMSprop(lr);

        return lossValue;
    }

    /**
     * Apply RMSprop optimization.
     */
    private void applyRMSprop(double learningRate) {
        double decay = config.getRmspropDecay();
        double epsilon = config.getRmspropEpsilon();
        double clipNorm = config.getGradientClipNorm();

        for (var pair : model.getParameters()) {
            String name = pair.getKey();
            var param = pair.getValue();

            if (param.requiresGradient()) {
                NDArray gradient = param.getArray().getGradient();

                if (gradient == null) {
                    continue;
                }

                // Gradient clipping
                if (clipNorm > 0) {
                    float gradNorm = gradient.norm().getFloat();
                    if (gradNorm > clipNorm) {
                        gradient = gradient.mul(clipNorm / gradNorm);
                    }
                }

                // RMSprop update
                NDArray sqGradAvg = squaredGradAvg.get(name);
                if (sqGradAvg == null) {
                    sqGradAvg = gradient.pow(2).mul(1 - decay);
                    squaredGradAvg.put(name, sqGradAvg);
                } else {
                    sqGradAvg = sqGradAvg.mul(decay).add(gradient.pow(2).mul(1 - decay));
                    squaredGradAvg.put(name, sqGradAvg);
                }

                NDArray update = gradient.div(sqGradAvg.sqrt().add(epsilon)).mul(learningRate);
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

            NDArray samples = sampler.generateSamples(
                    config.getNumSamplesToGenerate(),
                    parameterStore);

            // Convert from model output space back to image space
            samples = trainLoader.getPreprocessor().inversePreprocess(samples);

            // Save as image grid
            Path samplePath = Paths.get(config.getSampleDir(),
                    String.format("samples_epoch_%04d.png", epoch));
            ImageUtils.saveImageGrid(samples, samplePath.toString(), 8);

            logger.info("Samples saved to {}", samplePath);
        } catch (Exception e) {
            logger.error("Failed to generate samples at epoch {}", epoch, e);
        }
    }

    /**
     * Get current training state for resuming.
     */
    public TrainingState getState() {
        return new TrainingState(currentEpoch, globalStep, runningLoss, lossCount);
    }

    /**
     * Set training state for resuming.
     */
    public void setState(TrainingState state) {
        this.currentEpoch = state.epoch;
        this.globalStep = state.globalStep;
        this.runningLoss = state.runningLoss;
        this.lossCount = state.lossCount;
    }

    /**
     * Training state for checkpointing.
     */
    public static class TrainingState {
        public final int epoch;
        public final int globalStep;
        public final double runningLoss;
        public final int lossCount;

        public TrainingState(int epoch, int globalStep, double runningLoss, int lossCount) {
            this.epoch = epoch;
            this.globalStep = globalStep;
            this.runningLoss = runningLoss;
            this.lossCount = lossCount;
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
}
