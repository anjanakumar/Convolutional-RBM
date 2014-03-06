package crbm.rbm;

import crbm.ILogistic;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import java.util.Random;

/**
 * Created by Moritz on 2/25/14.
 */
public class SRBM implements IRBM {

    private final static Random RANDOM = new Random();
    private final float learnRate;
    private FloatMatrix weights;
    private final int desiredOutput;
    private final ILogistic logisticFunction;
    private float error = Float.POSITIVE_INFINITY;

    public SRBM(int input, float learnRate, int desiredOutput, ILogistic logisticFunction) {
        this.learnRate = learnRate;
        this.desiredOutput = desiredOutput;
        this.weights = FloatMatrix.rand(input, 2).mul(0.01f);
        this.logisticFunction = logisticFunction;
    }

    @Override
    public void train(float[][] data, StoppingCondition stop, boolean binarizeHidden, boolean binarizeVisible) {
        final FloatMatrix trainingData = new FloatMatrix(data);
        final FloatMatrix trainingDataTranspose = trainingData.transpose();
        final ForkBlas forkBlas = new ForkBlas();
        float lastError = 1.0f;
        while (stop.isNotDone()) {

            FloatMatrix hidden = new FloatMatrix(trainingData.getRows(), weights.getColumns());
            forkBlas.pmmuli(trainingData, weights, hidden);
            hidden = logisticFunction.function(hidden);

            FloatMatrix positiveA = new FloatMatrix(weights.getRows(), weights.getColumns());
            forkBlas.pmmuli(trainingDataTranspose, hidden, positiveA);

            FloatMatrix visible = new FloatMatrix(hidden.getRows(), weights.getRows());
            forkBlas.pmmuli(hidden, weights.transpose(), visible);
            visible = logisticFunction.function(visible);

            forkBlas.pmmuli(visible, weights, hidden);
            hidden = logisticFunction.function(hidden);

            FloatMatrix negativeA = new FloatMatrix(weights.getRows(), weights.getColumns());
            forkBlas.pmmuli(visible.transpose(), hidden, negativeA);

            weights.addi((positiveA.sub(negativeA)).div(trainingData.getRows()).mul(this.learnRate));

            error = (float) Math.sqrt(MatrixFunctions.pow(trainingData.sub(visible), 2.0f).sum() / trainingData.length / weights.getRows());

            float ratio = weights.getColumns() / (float) weights.getRows();
//            if (error > RANDOM.nextFloat()) {
//                weights = shuffleMonotoneColumns(weights);
//                lastError = 1.0f;
//                stop.setCurrentEpochs(0);
//                System.out.println("shuffle Column");
//            }
            //float threshold = (float) (0.01 * (Math.sqrt(ratio) + Math.pow(ratio, 4)));
            float threshold = 0.002f;
//            if (error < threshold && weights.getColumns() > desiredOutput) {
//                weights = shrink(weights);
//                lastError = 1.0f;
//                stop.setCurrentEpochs(0);
//            }
            stop.update(error);


            lastError = lastError * (1.0f - 0.005f/ weights.getColumns()) + (0.005f/ weights.getColumns()) * error;
            if (lastError - error < 1e-5f &&  error < threshold) {
                break;
            } else {
                if (error > threshold && lastError - error < 1e-5f) {
                    lastError = 1.0f;
                    weights = grow(weights);
                }
            }


            System.out.println("Remaining: " + (lastError - error));
            System.out.println("Error: " + error);
            System.out.println("rows: " + weights.getRows() + "  cols: " + weights.getColumns());
            System.out.println();
        }

    }

    private FloatMatrix shrink(FloatMatrix weights) {

        return shuffleSimilarRows(weights.get(new IntervalRange(0, weights.getRows()), new IntervalRange(0, weights.getColumns() - 1)));
    }

    private FloatMatrix grow(FloatMatrix weights) {
        FloatMatrix newColumn = new FloatMatrix(weights.getRows(), 1);
        float[] data = newColumn.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = 0.01f * (float) RANDOM.nextGaussian();
        }
        return FloatMatrix.concatHorizontally(shuffleSimilarColumns(weights), newColumn);
    }

    private FloatMatrix shuffleSimilarRows(FloatMatrix weights) {
        FloatMatrix result = weights.dup();
        float minDistance = Float.POSITIVE_INFINITY;
        int minDistanceIndex = 0;
        for (int i = 0; i < result.getRows(); i++) {
            FloatMatrix row1 = result.getRow(i);
            for (int j = i + 1; j < result.getRows(); j++) {
                float dist = (float) Math.sqrt(MatrixFunctions.pow(row1.sub(result.getRow(j)), 2).sum());
                if (dist < minDistance) {
                    minDistance = dist;
                    minDistanceIndex = j;
                }
            }
        }
        for (int i = 0; i < result.getColumns(); i++) {
            result.put(minDistanceIndex, i, 0.01f * (float) RANDOM.nextGaussian());
        }
        return result;
    }

    private FloatMatrix shuffleSimilarColumns(FloatMatrix weights) {
        FloatMatrix result = weights.dup();
        float minDistance = Float.POSITIVE_INFINITY;
        int minDistanceIndex = 0;
        for (int i = 0; i < result.getColumns(); i++) {
            FloatMatrix col1 = result.getColumn(i);
            for (int j = i + 1; j < result.getColumns(); j++) {
                float dist = (float) Math.sqrt(MatrixFunctions.pow(col1.sub(result.getColumn(j)), 2).sum());
                if (dist < minDistance) {
                    minDistance = dist;
                    minDistanceIndex = j;
                }
            }
        }
        for (int i = 0; i < result.getColumns(); i++) {
            result.put(i, minDistanceIndex, 0.01f * (float) RANDOM.nextGaussian());
        }
        return result;
    }

    @Override
    public float error(float[][] data, boolean binarizeHidden, boolean binarizeVisible) {
        return 0;
    }

    @Override
    public float[][] getHidden(float[][] data, boolean binarizeHidden) {
        return logisticFunction.function(new FloatMatrix(data).mmul(weights)).toArray2();
    }

    @Override
    public float[][] getVisible(float[][] data, boolean binarizeVisible) {

        final FloatMatrix dataMatrix = new FloatMatrix(data);
        return logisticFunction.function(dataMatrix.mmul(weights.transpose())).toArray2();

    }

    @Override
    public float[][] getWeights() {
        return this.weights.toArray2();
    }
}
