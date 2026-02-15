package com.technodrome.diffusion.training;

/**
 * Configuration parameters for diffusion model training.
 *
 * Contains all hyperparameters needed for training, including:
 * - Model architecture settings
 * - Optimization settings
 * - Training schedule
 * - Checkpointing settings
 */
public class TrainingConfig {

    // Model hyperparameters
    private int trajectoryLength = 1000;
    private int numTemporalBasis = 10;
    private double step1Beta = 1e-4;

    // Network architecture
    // Note: lowerHiddenChannels should be divisible by numScales
    private int lowerHiddenChannels = 126;  // 126 = 42 * 3 (divisible by numScales)
    private int upperHiddenChannels = 63;   // 63 = 21 * 3 (divisible by numScales)
    private int numLowerLayers = 4;
    private int numUpperLayers = 2;
    private int numScales = 3;
    private int kernelSize = 3;

    // Optimization
    private int batchSize = 512;
    private double learningRate = 1e-4;  // Reduced for stability
    private double learningRateDecayFactor = 0.1;
    private int learningRateDecayEpochs = 1000;
    private double rmspropDecay = 0.9;
    private double rmspropEpsilon = 1e-8;
    private double gradientClipNorm = 1.0;

    // Adaptive learning rate (reduce on plateau)
    private boolean useAdaptiveLR = false;
    private int lrPatience = 5;           // Epochs to wait before reducing LR
    private double lrReductionFactor = 0.5;  // Multiply LR by this when reducing
    private double minLearningRate = 1e-8;   // Don't reduce below this
    private double lrThreshold = 0.01;       // Minimum relative improvement to count as progress

    // Training schedule
    private int numEpochs = 2000;
    private int logEveryNSteps = 100;
    private int checkpointEveryNEpochs = 10;
    private int sampleEveryNEpochs = 10;

    // Data
    private int imageChannels = 1;
    private int imageHeight = 28;
    private int imageWidth = 28;

    // Paths
    private String checkpointDir = "checkpoints";
    private String sampleDir = "samples";
    private String logDir = "logs";

    // Sampling
    private int numSamplesToGenerate = 64;

    /**
     * Create default training configuration for MNIST.
     */
    public static TrainingConfig createMnistDefault() {
        return new TrainingConfig();
    }

    /**
     * Create a smaller configuration for testing/debugging.
     */
    public static TrainingConfig createSmallTest() {
        TrainingConfig config = new TrainingConfig();
        config.trajectoryLength = 100;
        config.numTemporalBasis = 5;
        config.lowerHiddenChannels = 32;
        config.upperHiddenChannels = 16;
        config.numLowerLayers = 2;
        config.numUpperLayers = 1;
        config.numScales = 2;
        config.batchSize = 64;
        config.numEpochs = 10;
        config.logEveryNSteps = 10;
        config.checkpointEveryNEpochs = 5;
        config.sampleEveryNEpochs = 5;
        return config;
    }

    /**
     * Get the current learning rate with decay applied.
     *
     * @param epoch Current epoch number
     * @return Decayed learning rate
     */
    public double getLearningRateAtEpoch(int epoch) {
        int numDecays = epoch / learningRateDecayEpochs;
        return learningRate * Math.pow(learningRateDecayFactor, numDecays);
    }

    // Builder-style setters

    public TrainingConfig setTrajectoryLength(int trajectoryLength) {
        this.trajectoryLength = trajectoryLength;
        return this;
    }

    public TrainingConfig setNumTemporalBasis(int numTemporalBasis) {
        this.numTemporalBasis = numTemporalBasis;
        return this;
    }

    public TrainingConfig setStep1Beta(double step1Beta) {
        this.step1Beta = step1Beta;
        return this;
    }

    public TrainingConfig setLowerHiddenChannels(int lowerHiddenChannels) {
        this.lowerHiddenChannels = lowerHiddenChannels;
        return this;
    }

    public TrainingConfig setUpperHiddenChannels(int upperHiddenChannels) {
        this.upperHiddenChannels = upperHiddenChannels;
        return this;
    }

    public TrainingConfig setNumLowerLayers(int numLowerLayers) {
        this.numLowerLayers = numLowerLayers;
        return this;
    }

    public TrainingConfig setNumUpperLayers(int numUpperLayers) {
        this.numUpperLayers = numUpperLayers;
        return this;
    }

    public TrainingConfig setNumScales(int numScales) {
        this.numScales = numScales;
        return this;
    }

    public TrainingConfig setKernelSize(int kernelSize) {
        this.kernelSize = kernelSize;
        return this;
    }

    public TrainingConfig setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public TrainingConfig setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public TrainingConfig setLearningRateDecayFactor(double learningRateDecayFactor) {
        this.learningRateDecayFactor = learningRateDecayFactor;
        return this;
    }

    public TrainingConfig setLearningRateDecayEpochs(int learningRateDecayEpochs) {
        this.learningRateDecayEpochs = learningRateDecayEpochs;
        return this;
    }

    public TrainingConfig setRmspropDecay(double rmspropDecay) {
        this.rmspropDecay = rmspropDecay;
        return this;
    }

    public TrainingConfig setGradientClipNorm(double gradientClipNorm) {
        this.gradientClipNorm = gradientClipNorm;
        return this;
    }

    public TrainingConfig setUseAdaptiveLR(boolean useAdaptiveLR) {
        this.useAdaptiveLR = useAdaptiveLR;
        return this;
    }

    public TrainingConfig setLrPatience(int lrPatience) {
        this.lrPatience = lrPatience;
        return this;
    }

    public TrainingConfig setLrReductionFactor(double lrReductionFactor) {
        this.lrReductionFactor = lrReductionFactor;
        return this;
    }

    public TrainingConfig setMinLearningRate(double minLearningRate) {
        this.minLearningRate = minLearningRate;
        return this;
    }

    public TrainingConfig setLrThreshold(double lrThreshold) {
        this.lrThreshold = lrThreshold;
        return this;
    }

    public TrainingConfig setNumEpochs(int numEpochs) {
        this.numEpochs = numEpochs;
        return this;
    }

    public TrainingConfig setLogEveryNSteps(int logEveryNSteps) {
        this.logEveryNSteps = logEveryNSteps;
        return this;
    }

    public TrainingConfig setCheckpointEveryNEpochs(int checkpointEveryNEpochs) {
        this.checkpointEveryNEpochs = checkpointEveryNEpochs;
        return this;
    }

    public TrainingConfig setSampleEveryNEpochs(int sampleEveryNEpochs) {
        this.sampleEveryNEpochs = sampleEveryNEpochs;
        return this;
    }

    public TrainingConfig setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
        return this;
    }

    public TrainingConfig setSampleDir(String sampleDir) {
        this.sampleDir = sampleDir;
        return this;
    }

    public TrainingConfig setLogDir(String logDir) {
        this.logDir = logDir;
        return this;
    }

    public TrainingConfig setNumSamplesToGenerate(int numSamplesToGenerate) {
        this.numSamplesToGenerate = numSamplesToGenerate;
        return this;
    }

    // Getters

    public int getTrajectoryLength() {
        return trajectoryLength;
    }

    public int getNumTemporalBasis() {
        return numTemporalBasis;
    }

    public double getStep1Beta() {
        return step1Beta;
    }

    public int getLowerHiddenChannels() {
        return lowerHiddenChannels;
    }

    public int getUpperHiddenChannels() {
        return upperHiddenChannels;
    }

    public int getNumLowerLayers() {
        return numLowerLayers;
    }

    public int getNumUpperLayers() {
        return numUpperLayers;
    }

    public int getNumScales() {
        return numScales;
    }

    public int getKernelSize() {
        return kernelSize;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getLearningRateDecayFactor() {
        return learningRateDecayFactor;
    }

    public int getLearningRateDecayEpochs() {
        return learningRateDecayEpochs;
    }

    public double getRmspropDecay() {
        return rmspropDecay;
    }

    public double getRmspropEpsilon() {
        return rmspropEpsilon;
    }

    public double getGradientClipNorm() {
        return gradientClipNorm;
    }

    public boolean isUseAdaptiveLR() {
        return useAdaptiveLR;
    }

    public int getLrPatience() {
        return lrPatience;
    }

    public double getLrReductionFactor() {
        return lrReductionFactor;
    }

    public double getMinLearningRate() {
        return minLearningRate;
    }

    public double getLrThreshold() {
        return lrThreshold;
    }

    public int getNumEpochs() {
        return numEpochs;
    }

    public int getLogEveryNSteps() {
        return logEveryNSteps;
    }

    public int getCheckpointEveryNEpochs() {
        return checkpointEveryNEpochs;
    }

    public int getSampleEveryNEpochs() {
        return sampleEveryNEpochs;
    }

    public int getImageChannels() {
        return imageChannels;
    }

    public int getImageHeight() {
        return imageHeight;
    }

    public int getImageWidth() {
        return imageWidth;
    }

    public String getCheckpointDir() {
        return checkpointDir;
    }

    public String getSampleDir() {
        return sampleDir;
    }

    public String getLogDir() {
        return logDir;
    }

    public int getNumSamplesToGenerate() {
        return numSamplesToGenerate;
    }

    @Override
    public String toString() {
        return "TrainingConfig{\n" +
                "  trajectoryLength=" + trajectoryLength + "\n" +
                "  numTemporalBasis=" + numTemporalBasis + "\n" +
                "  step1Beta=" + step1Beta + "\n" +
                "  batchSize=" + batchSize + "\n" +
                "  learningRate=" + learningRate + "\n" +
                "  numEpochs=" + numEpochs + "\n" +
                "  lowerHiddenChannels=" + lowerHiddenChannels + "\n" +
                "  upperHiddenChannels=" + upperHiddenChannels + "\n" +
                "  numLowerLayers=" + numLowerLayers + "\n" +
                "  numUpperLayers=" + numUpperLayers + "\n" +
                "  numScales=" + numScales + "\n" +
                "  useAdaptiveLR=" + useAdaptiveLR + "\n" +
                "  lrPatience=" + lrPatience + "\n" +
                "  lrReductionFactor=" + lrReductionFactor + "\n" +
                "  minLearningRate=" + minLearningRate + "\n" +
                "}";
    }
}
