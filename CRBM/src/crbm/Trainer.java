package crbm;

import java.util.Arrays;
import java.util.List;

public class Trainer {

    // data for training
    private static final String trainingDataPath = "CRBM/Data/MNIST_1000_Train_Database";
    // data for testing the generated clusters
    private static final String testDataPath = "CRBM/Data/MNIST_1000_Test_Database";

    // number of filters first CRBM
    private final int K1 = 15;
    // number of filters second CRBM
    private final int K2 = 15;
    // learning rate for updating weights
    private final float learningRate = 0.01f;
    // number of training epochs
    private final int epochs = 20;
    // data input size for first CRBM
    private final int crbm1DataEdgeLength = 32;
    // filter size CRBM
    private final int crbm1FilterEdgeLength = 5; // Output is 28x28
    private final int crbm2FilterEdgeLength = 5; // Output is 3x3
    // max pooling size after first CRBM
    private final int crbm1PoolingSize = 4;  // Output is 7x7
    private final int crbm2PoolingSize = 3;  // Output is 1x1

    public void train() {

        DataSet[] trainingDataSet = Main.loadData(trainingDataPath);
        float[][] trainingData = Main.dataSetToArray(trainingDataSet);
        
        // generate and train first CRBM
        CRBM crbm1 = new CRBM(K1, crbm1FilterEdgeLength, crbm1DataEdgeLength);
        crbm1.train(trainingData, epochs, learningRate);
        // get hidden propabilities for next CRBM input
        float[][][] hidden1 = crbm1.getHidden(trainingData);

        Main.exportAsImage(hidden1, "hidden1");

        // max pooling probabilities
        float[][][] hiddenMaxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbm1FilterEdgeLength);

        Main.exportAsImage(hiddenMaxPooled1, "hiddenMaxPooled1");

        // generate and train second CRBM
        CRBM crbm2 = new CRBM(K2, crbm1FilterEdgeLength, crbm1.getDataEdgeLength(hiddenMaxPooled1[0][0]));
        crbm2.train(hiddenMaxPooled1, epochs, learningRate);
        
        // get hidden probabilies
        float[][][][] hidden2 = crbm2.getHidden(hiddenMaxPooled1);

        Main.exportAsImage(hidden2, "hidden2");

        // max pooling probabilities
        float[][][][] hiddenMaxPooled2 = maxPooling(hidden2, crbm2PoolingSize, crbm2PoolingSize, crbm2FilterEdgeLength);

        Main.exportAsImage(hiddenMaxPooled2, "hiddenMaxPooled2");

        // generate feature vectors for each image from multidimensional data
        float[][] perceptron = new float[hiddenMaxPooled2.length][hiddenMaxPooled2[0].length * hiddenMaxPooled2[0][0].length];

        for(int i = 0; i < hiddenMaxPooled2.length; i++) {

            for(int pos = 0, k = 0; k < hiddenMaxPooled2[0].length; k++) {

                for(int kBefore = 0; kBefore < hiddenMaxPooled2[0][0].length; kBefore++, pos++) {

                    perceptron[i][pos] = hiddenMaxPooled2[i][k][kBefore][0];

                }

            }

        }

        // clustering with training data
        DataSet[] trainingDataResultSet = Main.arrayToDataSet(perceptron, trainingDataSet);
        List<Cluster> clusters = Main.generateClusters(trainingDataResultSet);
        Main.printClusters(clusters);
        
        // check clusters with test data and print results
        DataSet[] testDataSet = Main.loadData(testDataPath);
        float[][] testData = Main.dataSetToArray(testDataSet);
        float[][] testDataResult = getHiddenAll(testData, new CRBM[] {crbm1,crbm2});
        DataSet[] testDataResultSet = Main.arrayToDataSet(perceptron, testDataSet);
        Main.checkClusters(clusters, testDataResultSet);
    }

    /**
     * generates data feature vector matrix from multidimensional input
     * @param data
     * @return 
     */
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

    /**
     * nearest neighbour function is used by max pooling
     * when the data is not dividable by the max pooling size
     * @param pixels
     * @param actualWidth
     * @param actualHeight
     * @param newWidth
     * @param newHeight
     * @return 
     */
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

    /**
     * calls the real max pooling function
     * @param data
     * @param poolingSize
     * @param dataEdgeLength
     * @param filterEdgeLength
     * @return 
     */
    private float[][][][] maxPooling(float[][][][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        float[][][][] result = new float[data.length][][][];

        for(int i = 0; i < data.length; i++) {
            result[i] = maxPooling(data[i], poolingSize, dataEdgeLength , filterEdgeLength);
        }

        return result;
    }

    /**
     * calls the real max pooling function
     * @param data
     * @param poolingSize
     * @param dataEdgeLength
     * @param filterEdgeLength
     * @return 
     */
    private float[][][] maxPooling(float[][][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        float[][][] result = new float[data.length][][];

        for(int i = 0; i < data.length; i++) {
            result[i] = maxPooling(data[i], poolingSize, dataEdgeLength , filterEdgeLength);
        }

        return result;
    }

    /**
     * calculates the size of the data matrix after max pooling
     * @param poolingSize
     * @param dataEdgeLength
     * @param filterEdgeLength
     * @return 
     */
    private int maxPoolEdgeCalc(int poolingSize, int dataEdgeLength, int filterEdgeLength){

        int offset = filterEdgeLength-1;
        int pEdgeLength = dataEdgeLength - offset;

        if(pEdgeLength % poolingSize != 0){
            pEdgeLength = (pEdgeLength / poolingSize + 1) * poolingSize;

        }
        return pEdgeLength / poolingSize;
    }

    /**
     * max pooling reduces data matrix by a given factor poolingSize
     * picks only the max value in a pooling neighbourhood
     * @param data
     * @param poolingSize
     * @param dataEdgeLength
     * @param filterEdgeLength
     * @return 
     */
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

    // get hidden function for the complete CRBM stack
    private float[][] getHiddenAll(float[][] testData, CRBM[] crbms) {
        float[][][] hidden1 = crbms[0].getHidden(testData);

        float[][][] hiddenMaxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbm1FilterEdgeLength);

        float[][][][] hidden2 = crbms[1].getHidden(hiddenMaxPooled1);

        float[][][][] hiddenMaxPooled2 = maxPooling(hidden2, crbm2PoolingSize, crbm2PoolingSize, crbm2FilterEdgeLength);

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
}
