package com.technodrome.diffusion.data;

import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.Iterator;

/**
 * MNIST data loader using DJL's built-in dataset.
 *
 * Provides:
 * - Automatic download and caching of MNIST data
 * - Batching with configurable batch size
 * - Preprocessing with z-score normalization
 * - Optional uniform noise for dequantization
 */
public class MnistDataLoader {

    private final int batchSize;
    private final boolean training;
    private final DataPreprocessor preprocessor;
    private Mnist dataset;

    /**
     * Create an MNIST data loader.
     *
     * @param batchSize Batch size
     * @param training Whether to use training set (true) or test set (false)
     */
    public MnistDataLoader(int batchSize, boolean training) {
        this.batchSize = batchSize;
        this.training = training;
        this.preprocessor = DataPreprocessor.createMnistPreprocessor();
    }

    /**
     * Create an MNIST data loader with custom preprocessor.
     */
    public MnistDataLoader(int batchSize, boolean training, DataPreprocessor preprocessor) {
        this.batchSize = batchSize;
        this.training = training;
        this.preprocessor = preprocessor;
    }

    /**
     * Prepare the dataset (download if necessary).
     */
    public void prepare(NDManager manager) throws IOException, TranslateException {
        Mnist.Builder builder = Mnist.builder()
                .setSampling(batchSize, true)  // shuffle for training
                .optManager(manager);

        if (training) {
            builder.optUsage(Dataset.Usage.TRAIN);
        } else {
            builder.optUsage(Dataset.Usage.TEST);
        }

        this.dataset = builder.build();
        this.dataset.prepare();
    }

    /**
     * Get an iterator over batches.
     */
    public Iterable<Batch> iterateBatches(NDManager manager) throws IOException, TranslateException {
        if (dataset == null) {
            prepare(manager);
        }
        return dataset.getData(manager);
    }

    /**
     * Get a single batch of preprocessed images.
     *
     * @param manager NDManager
     * @return Preprocessed images of shape (batchSize, 1, 28, 28)
     */
    public NDArray getBatch(NDManager manager) throws IOException, TranslateException {
        if (dataset == null) {
            prepare(manager);
        }

        Iterator<Batch> iterator = dataset.getData(manager).iterator();
        if (iterator.hasNext()) {
            Batch batch = iterator.next();
            NDArray images = batch.getData().singletonOrThrow();
            return preprocessor.preprocess(images);
        }
        throw new IllegalStateException("No data available");
    }

    /**
     * Get number of batches per epoch.
     */
    public long getNumBatches() {
        if (dataset == null) {
            throw new IllegalStateException("Dataset not prepared");
        }
        return (dataset.size() + batchSize - 1) / batchSize;
    }

    /**
     * Get total number of samples.
     */
    public long getNumSamples() {
        if (dataset == null) {
            throw new IllegalStateException("Dataset not prepared");
        }
        return dataset.size();
    }

    /**
     * Preprocess a batch from the dataset.
     */
    public NDArray preprocessBatch(Batch batch) {
        NDArray images = batch.getData().singletonOrThrow();
        return preprocessor.preprocess(images);
    }

    /**
     * Get raw batch without preprocessing.
     */
    public NDArray getRawBatch(Batch batch) {
        return batch.getData().singletonOrThrow();
    }

    /**
     * Create batched iterator with preprocessing.
     */
    public Iterable<NDArray> iteratePreprocessedBatches(NDManager manager) throws IOException, TranslateException {
        return () -> {
            try {
                Iterator<Batch> batchIterator = iterateBatches(manager).iterator();
                return new Iterator<NDArray>() {
                    @Override
                    public boolean hasNext() {
                        return batchIterator.hasNext();
                    }

                    @Override
                    public NDArray next() {
                        Batch batch = batchIterator.next();
                        try {
                            return preprocessBatch(batch);
                        } finally {
                            batch.close();
                        }
                    }
                };
            } catch (Exception e) {
                throw new RuntimeException("Error creating batch iterator", e);
            }
        };
    }

    /**
     * Load entire dataset into memory (for small datasets like MNIST).
     */
    public NDArray loadAllData(NDManager manager) throws IOException, TranslateException {
        if (dataset == null) {
            prepare(manager);
        }

        // Collect all batches
        java.util.List<NDArray> batches = new java.util.ArrayList<>();
        for (Batch batch : dataset.getData(manager)) {
            batches.add(batch.getData().singletonOrThrow().duplicate());
            batch.close();
        }

        // Concatenate all batches
        NDArray allData = batches.get(0);
        for (int i = 1; i < batches.size(); i++) {
            allData = allData.concat(batches.get(i), 0);
        }

        return preprocessor.preprocess(allData);
    }

    // Getters
    public int getBatchSize() {
        return batchSize;
    }

    public boolean isTraining() {
        return training;
    }

    public DataPreprocessor getPreprocessor() {
        return preprocessor;
    }

    public Mnist getDataset() {
        return dataset;
    }
}
