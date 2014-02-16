package crbm;

import crbm.rbm.IRBM;
import crbm.rbm.RBMJBlasOpti;
import crbm.rbm.StoppingCondition;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Radek on 08.02.14.
 */
public class Trainer {

    private static final String trainingDataPath = "CRBM/Data/MNIST_1000_Database";
    private static final String testDataPath = "CRBM/Data/MNIST_1000_Database";


    private final int K1 = 15;
    private final int K2 = 15;
    private final float learningRate = 0.01f;
    private final int epochs = 3;

    private final int crbmFilterEdgeLength = 5;

    private final int crbm1DataEdgeLength = 32;
    private final int crbm1PoolingSize = 2;

    private final int crbm2DataEdgeLength = crbm1DataEdgeLength - crbmFilterEdgeLength + 1;
    private final int crbm2PoolingSize = 2;

    private final int rbmOutputSize = 30;

    public void train() {

        DataSet[] trainingDataSet = Main.loadData(trainingDataPath);
        float[][] trainingData = Main.dataSetToArray(trainingDataSet);

        CRBM crbm1 = new CRBM(K1, crbmFilterEdgeLength, crbm1DataEdgeLength);
        crbm1.train(trainingData, epochs, learningRate);
        float[][][] hidden1 = crbm1.getHidden(trainingData);

        Main.exportAsImage(hidden1, "hidden1");

        float[][][] hiddenMaxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);

        Main.exportAsImage(hiddenMaxPooled1, "hiddenMaxPooled1");

        CRBM crbm2 = new CRBM(K2, crbmFilterEdgeLength, crbm1.getDataEdgeLength(hiddenMaxPooled1[0][0]));
        crbm2.train(hiddenMaxPooled1, epochs, learningRate);
        float[][][][] hidden2 = crbm2.getHidden(hiddenMaxPooled1);

        Main.exportAsImage(hidden2, "hidden2");

        float[][][][] hiddenMaxPooled2 = maxPooling(hidden2, crbm1PoolingSize, 5, crbmFilterEdgeLength);

        Main.exportAsImage(hiddenMaxPooled2, "hiddenMaxPooled2");

        float[][] perceptron = new float[hiddenMaxPooled2.length][hiddenMaxPooled2[0].length * hiddenMaxPooled2[0].length];

        for(int i = 0; i < hiddenMaxPooled2.length; i++) {

            for(int pos = 0, k = 0; k < hiddenMaxPooled2[0].length; k++) {

                for(int kBefore = 0; kBefore < hiddenMaxPooled2[0][0].length; kBefore++, pos++) {

                    perceptron[i][pos] = hiddenMaxPooled2[i][k][kBefore][0];

                }

            }

        }


        /*
        // Use plain old RBM        
        float[][] rbmData = concatDataPartitions(hiddenMaxPooled1);

        IRBM rbm = new RBMJBlasOpti(rbmData[0].length, 100, learningRate, new DefaultLogisticMatrixFunction(), false, 0, null);
        rbm.train(rbmData,new StoppingCondition(2000), false, false);
        float[][] trainingDataResult = rbm.getHidden(rbmData, false);
        */

        // Clustering
        DataSet[] trainingDataResultSet = Main.arrayToDataSet(perceptron, trainingDataSet);
        List<Cluster> clusters = Main.generateClusters(trainingDataResultSet);
        Main.printClusters(clusters);
        
        // Check Clusters
        DataSet[] testDataSet = Main.loadData(testDataPath);
        float[][] testData = Main.dataSetToArray(testDataSet);
        float[][] testDataResult = getHiddenAll(testData, new CRBM[] {crbm1,crbm2});
        DataSet[] testDataResultSet = Main.arrayToDataSet(perceptron, testDataSet);
        Main.checkClusters(clusters, testDataResultSet);
    }

    private float[][] concatDataPartitions(float[][][] data){
        float[][] result = new float[data.length][data[0].length * data[0][0].length];

        for(int i = 0; i < data.length; ++i){
            for(int j = 0; j < data[0].length; ++j){
                for(int k = 0; k < data[0][0].length; ++k){
                    result[i][j * data[0][0].length + k] = data[i][j][k];
                }
            }
        }

        return result;
    }

    public float[] nearestNeighbour(float[] pixels, int actualWidth, int actualHeight, int newWidth, int newHeight) {

        float[] result = new float[newWidth * newHeight];
        float x_ratio = actualWidth / (float) newWidth;
        float y_ratio = actualHeight / (float) newHeight;
        float px, py;

        for (int i = 0; i < newHeight; i++) {
            for (int j = 0; j < newWidth; j++) {
                px = (int)(j * x_ratio);
                py = (int)(i * y_ratio);
                result[(i * newWidth) + j] = pixels[(int) ((py * actualWidth) + px)];
            }
        }

        return result;
    }

    private float[][][][] maxPooling(float[][][][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        float[][][][] result = new float[data.length][][][];

        for(int i = 0; i < data.length; i++) {
            result[i] = maxPooling(data[i], poolingSize, dataEdgeLength , filterEdgeLength);
        }

        return result;
    }

    private float[][][] maxPooling(float[][][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        float[][][] result = new float[data.length][][];

        for(int i = 0; i < data.length; i++) {
            result[i] = maxPooling(data[i], poolingSize, dataEdgeLength , filterEdgeLength);
        }

        return result;
    }

    private int maxPoolEdgeCalc(int poolingSize, int dataEdgeLength, int filterEdgeLength){

        int offset = filterEdgeLength-1;
        int pEdgeLength = dataEdgeLength - offset;

        if(pEdgeLength % poolingSize != 0){
            pEdgeLength = (pEdgeLength / poolingSize + 1) * poolingSize;

        }
        return pEdgeLength / poolingSize;
    }

    private float[][] maxPooling(float[][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        int offset = filterEdgeLength-1;
        int pEdgeLength = dataEdgeLength - offset;
        int iEdgeLength = dataEdgeLength - offset;

        float[][] workingData = new float[data.length][];
        if(iEdgeLength % poolingSize != 0){
            pEdgeLength = (iEdgeLength / poolingSize + 1) * poolingSize;
            for(int i = 0; i < data.length; ++i){
                workingData[i] = nearestNeighbour(data[i], iEdgeLength, iEdgeLength, pEdgeLength, pEdgeLength);
            }
        } else{
            for(int i = 0; i < data.length; ++i){
                workingData[i] = Arrays.copyOf(data[i], iEdgeLength * iEdgeLength);
            }
        }

        int rEdgeLength = pEdgeLength / poolingSize;
        float[][] result = new float[K1][rEdgeLength * rEdgeLength];

        for(int k = 0; k < K1; k++) {

            for (int y = 0; y < rEdgeLength; y++) {
                for (int x = 0; x < rEdgeLength; x++) {

                    float max = 0;
                    for (int yh = 0; yh < poolingSize; yh++) {
                        for (int xh = 0; xh < poolingSize; xh++) {

                            int pos = (y * poolingSize + yh) * pEdgeLength + x*poolingSize + xh;
                            float value = workingData[k][pos];

                            max = Math.max(value, max);

                        }
                    }

                    result[k][y * rEdgeLength + x] = max;
                }
            }
        }

        return result;
    }



    private float[][][] normalize(float[][][] data) {
        float[][][] result = new float[data.length][][];

        for (int i = 0; i < result.length; i++) {
            result[i] = normalize(data[i]);
        }
        return result;
    }

    private float[][] normalize(float[][] data) {
        float[][] result = new float[data.length][];

        for (int i = 0; i < result.length; i++) {
            result[i] = normalize(data[i]);
        }
        return result;
    }

    private float[] normalize(float[] data) {
        float[] result = new float[data.length];

        float max = Float.NEGATIVE_INFINITY;
        float min = Float.POSITIVE_INFINITY;

        for (float f : data) {
            if(f > max) max = f;
            if(f < min) min = f;
        }

        float range = max - min;
        for (int i = 0; i < data.length; i++) {
            result[i] = (data[i] - min)/range;
        }
        return result;
    }

    private float[][] getHiddenAll(float[][] testData, CRBM[] crbms) {
        float[][][] hidden1 = crbms[0].getHidden(testData);

        float[][][] hiddenMaxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);

        float[][][][] hidden2 = crbms[1].getHidden(hiddenMaxPooled1);

        float[][][][] hiddenMaxPooled2 = maxPooling(hidden2, crbm1PoolingSize, 5, crbmFilterEdgeLength);

        float[][] perceptron = new float[hiddenMaxPooled2.length][hiddenMaxPooled2[0].length * hiddenMaxPooled2[0].length];

        for(int i = 0; i < hiddenMaxPooled2.length; i++) {

            for(int pos = 0, k = 0; k < hiddenMaxPooled2[0].length; k++) {

                for(int kBefore = 0; kBefore < hiddenMaxPooled2[0][0].length; kBefore++, pos++) {

                    perceptron[i][pos] = hiddenMaxPooled2[i][k][kBefore][0];

                }

            }

        }

        return perceptron;
    }

    private float[][] copyArray2D(float[][] arrays) {
        float[][] result = new float[arrays.length][];

        for (int i = 0; i < arrays.length; i++) {
            result[i] = Arrays.copyOf(arrays[i], arrays[i].length);
        }

        return result;
    }
}
