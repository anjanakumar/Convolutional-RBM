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
    
    private static final String trainingDataPath = "Data/MNIST_Small";
    private static final String testDataPath = "Data/MNIST_Small";
    
    private final int K = 15;
    private final float learningRate = 0.01f;
    private final int epochs = 100;

    private final int crbmFilterEdgeLength = 5;

    private final int crbm1DataEdgeLength = 32;
    private final int crbm1PoolingSize = 2;

    private final int crbm2DataEdgeLength = crbm1DataEdgeLength - crbmFilterEdgeLength + 1;
    private final int crbm2PoolingSize = 2;
    
    private final int rbmOutputSize = 30;

    public void train() {

        DataSet[] trainingDataSet = Main.loadData(trainingDataPath);
        float[][] trainingData = Main.dataSetToArray(trainingDataSet);

        CRBM crbm1 = new CRBM(K, crbmFilterEdgeLength);
        crbm1.train(trainingData, crbm1DataEdgeLength, epochs, learningRate, "First-RBM");
        float[][][] hidden1 = crbm1.getHidden(trainingData, crbm1DataEdgeLength);

        Main.exportAsImage(reduceDimension(hidden1), "hidden1", 0);

        float[][][] maxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);
        int crbm2MaxPooledDataEdgeLength = maxPoolEdgeCalc(crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);

        Main.exportAsImage(reduceDimension(maxPooled1), "maxPooled1", 0);

        CRBM crbm2 = new CRBM(K, crbmFilterEdgeLength);
        crbm2.train3D(maxPooled1, crbm2MaxPooledDataEdgeLength, epochs, learningRate, "Second-RBM");
        float[][][] hidden2 = crbm2.getHidden3D(maxPooled1, crbm2MaxPooledDataEdgeLength);

        Main.exportAsImage(reduceDimension(hidden2), "hidden2", 0);

        float[][][] maxPooled2 = maxPooling(hidden2, crbm2PoolingSize, crbm2MaxPooledDataEdgeLength, crbmFilterEdgeLength);

        // reduce for export
        float[][] reducedMaxPooled2 = reduceDimension(maxPooled2);
        Main.exportAsImage(reducedMaxPooled2, "maxPooled2", 0);
       
        float[][] rbmData = concatDataPartitions(maxPooled2);
        
        IRBM rbm = new RBMJBlasOpti(rbmData[0].length, 100, learningRate, new DefaultLogisticMatrixFunction(), false, 0, null);
        rbm.train(rbmData,new StoppingCondition(2000), false, false);
        float[][] trainingDataResult = rbm.getHidden(rbmData, false);
        
        // Clustering
        DataSet[] trainingDataResultSet = Main.arrayToDataSet(trainingDataResult, trainingDataSet);
        List<Cluster> clusters = Main.generateClusters(trainingDataResultSet);
        Main.printClusters(clusters);
        
        // Check Clusters
        DataSet[] testDataSet = Main.loadData(testDataPath);
        float[][] testData = Main.dataSetToArray(testDataSet);
        float[][] testDataResult = getHiddenAll(testData, new CRBM[] {crbm1, crbm2}, new IRBM[]{rbm});
        DataSet[] testDataResultSet = Main.arrayToDataSet(testDataResult, testDataSet);
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

    float[][][] expandDimension(float[][] data, int nextDimensionSize) {
        float[][][] result = new float[data.length / nextDimensionSize][nextDimensionSize][data[0].length];

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                for(int k = 0; k < result[0][0].length; k++) {
                    result[i][j][k] = data[i * result[0].length + j][k];
                }
            }
        }
        return result;
    }

    float[][] reduceDimension(float[][][] data) {
        float[][] result = new float[data.length * data[0].length][];

        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < data[0].length; j++){
                result[i * data[0].length + j] = data[i][j];
            }
        }
        return result;
    }

    public float[][][] nearestNeighbour(float[][][] pixels, int actualWidth, int actualHeight, int newWidth, int newHeight) {
        float[][][] result = new float[pixels.length][pixels[0].length][];
        for(int i = 0; i < pixels.length; i++) {
            result[i] = nearestNeighbour(pixels[i], actualWidth, actualHeight, newWidth, newHeight);
        }

        return result;
    }

    public float[][] nearestNeighbour(float[][] pixels, int actualWidth, int actualHeight, int newWidth, int newHeight) {
        float[][] result = new float[pixels.length][];

        for(int i = 0; i < pixels.length; i++) {
            result[i] = nearestNeighbour(pixels[i], actualWidth, actualHeight, newWidth, newHeight);
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

    private float[][][] maxPooling(float[][][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        float[][][] result = new float[data.length][K][];

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
        float[][] result = new float[K][rEdgeLength * rEdgeLength];

        for(int k = 0; k < K; k++) {

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
        
        for(int i = 0; i < rEdgeLength; ++i){
            for(int j = 0; j < rEdgeLength; ++j){
                int pos = i * rEdgeLength + j;
                boolean foundOnce = false;
                boolean foundTwice = false;
                int foundOncePosition = 0;
                for(int k = 0; k < K; ++k){
                    if(foundTwice){
                        result[k][pos] = 0.0f;
                    }else if(foundOnce){
                        if(result[k][pos] > 0.5f){
                            foundTwice = true;
                            result[k][foundOncePosition] = 0.0f;
                        }
                        result[k][pos] = 0.0f;
                    }else if(result[k][pos] > 0.5f){
                        foundOnce = true;
                        foundOncePosition = pos;
                        result[k][pos] = 1.0f;
                    }else{
                        result[k][pos] = 0.0f;
                    }
                }
            }
        }

        return result;
    }

    private float[][] getHiddenAll(float[][] testData, CRBM[] crbms, IRBM[] irbms) {
        float[][][] hidden1 = crbms[0].getHidden(testData, crbm1DataEdgeLength);

        float[][][] maxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);
        int crbm2MaxPooledDataEdgeLength = maxPoolEdgeCalc(crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);
        float[][][] hidden2 = crbms[1].getHidden3D(maxPooled1, crbm2MaxPooledDataEdgeLength);
        float[][][] maxPooled2 = maxPooling(hidden2, crbm2PoolingSize, crbm2MaxPooledDataEdgeLength, crbmFilterEdgeLength);
         
        float[][] rbmData = concatDataPartitions(maxPooled2);
        
        float[][] hidden = copyArray2D(rbmData);
        
        for (IRBM rbm : irbms) {
            hidden = rbm.getHidden(hidden, true);
        }
        return hidden;
    }
    
    private float[][] copyArray2D(float[][] arrays) {
        float[][] result = new float[arrays.length][];
        
        for (int i = 0; i < arrays.length; i++) {
            result[i] = Arrays.copyOf(arrays[i], arrays[i].length);
        }
        
        return result;
    }
}
