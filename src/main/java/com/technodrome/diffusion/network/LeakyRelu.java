package com.technodrome.diffusion.network;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Leaky ReLU activation function.
 *
 * f(x) = x if x >= 0
 *      = slope * x if x < 0
 *
 * Default slope of 0.05 as used in the original diffusion model paper.
 */
public class LeakyRelu extends AbstractBlock {

    private static final byte VERSION = 1;

    private final float slope;

    /**
     * Create a Leaky ReLU with default slope (0.05).
     */
    public LeakyRelu() {
        this(0.05f);
    }

    /**
     * Create a Leaky ReLU with specified slope.
     *
     * @param slope Slope for negative values (typically small, e.g., 0.01-0.1)
     */
    public LeakyRelu(float slope) {
        super(VERSION);
        this.slope = slope;
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        NDArray result = leakyRelu(x);
        return new NDList(result);
    }

    /**
     * Apply leaky ReLU to an array.
     */
    public NDArray leakyRelu(NDArray x) {
        // f(x) = max(x, 0) + slope * min(x, 0)
        // Equivalent to: x.maximum(0) + slope * x.minimum(0)
        NDArray positive = x.maximum(0);
        NDArray negative = x.minimum(0).mul(slope);
        return positive.add(negative);
    }

    /**
     * Static utility method for applying leaky ReLU.
     */
    public static NDArray apply(NDArray x, float slope) {
        NDArray positive = x.maximum(0);
        NDArray negative = x.minimum(0).mul(slope);
        return positive.add(negative);
    }

    /**
     * Static utility method with default slope.
     */
    public static NDArray apply(NDArray x) {
        return apply(x, 0.05f);
    }

    @Override
    public ai.djl.ndarray.types.Shape[] getOutputShapes(ai.djl.ndarray.types.Shape[] inputShapes) {
        return inputShapes;
    }

    public float getSlope() {
        return slope;
    }
}
