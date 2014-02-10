package crbm;

import crbm.rbm.IRBM;
import crbm.rbm.RBMJBlasOpti;
import crbm.rbm.StoppingCondition;
import java.util.Arrays;

/**
 * Created by Radek on 08.02.14.
 */
public class Trainer {
    
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

        float[][] data = Main.loadData();

        CRBM crbm1 = new CRBM(K, crbmFilterEdgeLength);
        crbm1.train(data, crbm1DataEdgeLength, epochs, learningRate, "First-RBM");
        float[][][] hidden1 = crbm1.getHidden(data, crbm1DataEdgeLength);

        // EXPORT
        Main.exportAsImage(reduceDimension(hidden1), "hidden1");
        // EXPORT END

        float[][][] maxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);
        int crbm2MaxPooledDataEdgeLength = maxPoolEdgeCalc(crbm1PoolingSize, crbm1DataEdgeLength, crbmFilterEdgeLength);

        // EXPORT
        Main.exportAsImage(reduceDimension(maxPooled1), "maxPooled1");
        // EXPORT END

        CRBM crbm2 = new CRBM(K, crbmFilterEdgeLength);
        crbm2.train(reduceDimension(maxPooled1), crbm2MaxPooledDataEdgeLength, epochs, learningRate, "Second-RBM");
        float[][][] hidden2 = crbm2.getHidden(reduceDimension(maxPooled1), crbm2MaxPooledDataEdgeLength);

        // EXPORT
        Main.exportAsImage(reduceDimension(hidden2), "hidden2");
        // EXPORT END

        float[][][] maxPooled2 = maxPooling(hidden2, crbm2PoolingSize, crbm2MaxPooledDataEdgeLength, crbmFilterEdgeLength);

        float[][] features = reduceDimension(maxPooled2);
        
        // End Training
        // Reconstructions

        // EXPORT
        Main.exportAsImage(features, "maxPooled2");
        // EXPORT END

        CRBM crbm3 = new CRBM(K, crbmFilterEdgeLength);
        crbm3.train(reduceDimension(hidden1), crbm2DataEdgeLength, epochs, learningRate, "Second-RBM");
        float[][][] hidden3 = crbm3.getHidden(reduceDimension(hidden1), crbm2DataEdgeLength);

        // EXPORT
        Main.exportAsImage(reduceDimension(hidden3), "hidden3");
        // EXPORT END

        float[][] visible2 = crbm3.getVisible(hidden3, null, crbm2DataEdgeLength);
        Main.exportAsImage(visible2, "visible2");

        float[][] visible1 = crbm1.getVisible(expandDimension(visible2, K), null, crbm2DataEdgeLength - crbmFilterEdgeLength + 1);

        Main.exportAsImage(visible1, "visible1");
        
        // Use plain old RBM
        IRBM rbm = new RBMJBlasOpti(15, 10, learningRate, new DefaultLogisticMatrixFunction(), false, 0, null);;
        float[][] rbmData = new float[][]{};
        rbm.train(rbmData,new StoppingCondition(epochs), false, false);
        float[][] rbmResult = rbm.getHidden(rbmData, false);
        
        // Use Generalized Lloyd for final clustering
        Cluster[] clusters = Main.clustering(rbmResult);
        Main.printClusters(clusters);
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

        return result;
    }
}
