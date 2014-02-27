package crbm.rbm;

import com.sun.java.swing.plaf.windows.resources.windows;
import crbm.ILogistic;
import crbm.rbm.ForkBlas;
import crbm.rbm.IRBM;
import crbm.rbm.StoppingCondition;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import java.util.Random;

/**
 * Created by Moritz on 2/25/14.
 */
public class SRBM implements IRBM{

    private final static Random RANDOM = new Random();
    private final float learnRate;
    private FloatMatrix weights;
    private final int desiredOutput;
    private final ILogistic logisticFunction;
    private float error = Float.POSITIVE_INFINITY;

    public SRBM(int input, float learnRate, int desiredOutput, ILogistic logisticFunction) {
        this.learnRate = learnRate;
        this.desiredOutput = desiredOutput;
        this.weights = FloatMatrix.rand(input,input).mul(0.01f);
        this.logisticFunction = logisticFunction;
    }

    @Override
    public void train(float[][] data, StoppingCondition stop, boolean binarizeHidden, boolean binarizeVisible) {
        final FloatMatrix trainingData = new FloatMatrix(data);
        final FloatMatrix trainingDataTranspose = trainingData.transpose();
        final ForkBlas forkBlas = new ForkBlas();
        float lastError = 1.0f;
        while(stop.isNotDone()) {

            FloatMatrix hidden = new FloatMatrix(trainingData.getRows(), weights.getColumns());
            forkBlas.pmmuli(trainingData, weights, hidden);
            hidden =  logisticFunction.function(hidden);

            FloatMatrix positiveA = new FloatMatrix(weights.getRows(), weights.getColumns());
            forkBlas.pmmuli(trainingDataTranspose, hidden, positiveA);

            FloatMatrix visible = new FloatMatrix(hidden.getRows(), weights.getRows());
            forkBlas.pmmuli(hidden, weights.transpose(), visible);
            visible = logisticFunction.function(visible);

            forkBlas.pmmuli(visible, weights, hidden);
            hidden =  logisticFunction.function(hidden);

            FloatMatrix negativeA = new FloatMatrix(weights.getRows(), weights.getColumns());
            forkBlas.pmmuli(visible.transpose(),hidden, negativeA);

            weights.addi((positiveA.sub(negativeA)).div(trainingData.getRows()).mul(this.learnRate));

            error = (float)Math.sqrt(MatrixFunctions.pow(trainingData.sub(visible), 2.0f).sum() / trainingData.length / weights.getRows());

            if(error < 0.01f * Math.cbrt(weights.getColumns()/ (float) weights.getRows()) && weights.getColumns() > desiredOutput) {
                weights = shrink(weights);
                lastError = 1.0f;
                stop.setCurrentEpochs(0);
            }
            stop.update(error);


            System.out.println("Remaining: " + (lastError - error));
            System.out.println("Error: " + error);
            System.out.println("rows: "  + weights.getRows() + "  cols: " + weights.getColumns());

            System.out.println();
            lastError = lastError * 0.99f + 0.01f * error;
            if(lastError - error < 1e-5f) break;
        }

    }

    private FloatMatrix shrink(FloatMatrix weights) {
        FloatMatrix result = weights.get(new IntervalRange(0, weights.getRows()), new IntervalRange(0, weights.getColumns() - 1));

        float minDistance = Float.POSITIVE_INFINITY;

        int minDistanceIndex = 0;
        for(int i = 0; i < result.getRows(); i++) {
            FloatMatrix row1 = result.getRow(i);
            for (int j = i + 1; j < result.getRows(); j++) {
                float dist =  (float)Math.sqrt(MatrixFunctions.pow(row1.sub(result.getRow(j)), 2).sum());
                if(dist < minDistance) {
                    minDistance = dist;
                    minDistanceIndex = j;
                }
            }
        }
        for(int i = 0; i < result.getColumns(); i++) {
            result.put(minDistanceIndex, i, 0.01f * (float)RANDOM.nextGaussian());
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
