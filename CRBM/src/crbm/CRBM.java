/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package crbm;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.FileUtils;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import javax.imageio.ImageIO;

/**
 *
 * @author Radek
 */
public class CRBM {

    private final Random RANDOM = new Random();

    private final float[][] W;
    private final int filterEdgeLength;

    private float c_k = 0f;
    private float b = 0f;


    public CRBM(int K, int filterEdgeLength) {
        this.W = initW(K, filterEdgeLength);
        this.filterEdgeLength = filterEdgeLength;
    }

    private float[][] initW(int K, int filterEdgeLength) {
        int filterDimensions = filterEdgeLength * filterEdgeLength;
        float[][] result = new float[K][filterDimensions];

        for (int k = 0; k < K; k++) {
            for (int i = 0; i < filterDimensions; i++) {
                result[k][i] = (float) (((RANDOM.nextDouble() - 0.5) * 2) / (filterEdgeLength * filterEdgeLength));
            }
        }
        return result;
    }

    public void train(float[][][] data, int dataEdgeLength, int epochs, float learningRate, String exportPath) {

        for (int e = 0; e < epochs; e++) {
            float error = 0;
            for (float[][] aData : data) {
                error += train(aData, dataEdgeLength, learningRate);
            }
            error /= data.length;
            System.out.println(error);
        }
        //killFirst();

        System.out.println("Training finished");
    }

    public void train(float[][] data, int dataEdgeLength, int epochs, float learningRate, String exportPath) {

        for (int e = 0; e < epochs; e++) {
            float error = 0;
            for (float[] aData : data) {
                error += train(aData, dataEdgeLength, learningRate);
            }
            error /= data.length;
            System.out.println(error);
        }
        //killFirst();

        System.out.println("Training finished");
    }

    public void killFirst() {
        for(int i = 0; i < W[0].length; i++) {
            W[0][i] = 0;
        }
    }

    private float train(float[][] data, int dataEdgeLength, float learningRate) {
        int offset = filterEdgeLength - 1;
        int K = W.length;
        float[][] W1;

        float[][] PH0_k = new float[K][];
        float[][] Grad0_k = new float[K][];
        float[][] H0_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH0_k[k] = filter(data, W[k], dataEdgeLength, filterEdgeLength, true);
            if(k != 0) {
                addBias(PH0_k[k], c_k);
            }
            PH0_k[k] = logistic(PH0_k[k]);
            Grad0_k[k] = filter(data, PH0_k[k], dataEdgeLength, dataEdgeLength - offset, true);
            H0_k[k] = bernoulli(PH0_k[k]);
        }

        W1 = flip(W);

        float[] V1m = new float[(dataEdgeLength - offset * 2) * (dataEdgeLength - offset * 2)];
        for (int k = 0; k < W1.length; k++) {
            float[] r = filter(data, W1[k], dataEdgeLength - offset, this.filterEdgeLength, true);
            V1m = add(V1m, r);
        }

        addBias(V1m, b);
        float[] V1 = logistic(V1m);
        if(data != null) {
            V1 = concat(data, V1, dataEdgeLength, filterEdgeLength);
        }

        // float[] V1 = getVisible(H0_k, data, dataEdgeLength);

        float[][] PH1_k = new float[K][];
        float[][] Grad1_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH1_k[k] = filter(V1, W[k], dataEdgeLength, filterEdgeLength);
            if(k != 0) {
                addBias(PH0_k[k], c_k);
            }
            PH1_k[k] = logistic(PH1_k[k]);
            Grad1_k[k] = filter(V1, PH1_k[k], dataEdgeLength, dataEdgeLength - offset);
            W[k] = CD(W[k], Grad0_k[k], Grad1_k[k], learningRate);
        }

        return 0.0f;
    }

    private float train(float[] data, int dataEdgeLength, float learningRate) {
        int offset = filterEdgeLength - 1;
        int K = W.length;
        float[][] W1;

        float[][] PH0_k = new float[K][];
        float[][] Grad0_k = new float[K][];
        float[][] H0_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH0_k[k] = filter(data, W[k], dataEdgeLength, filterEdgeLength);
            if(k != 0) {
                addBias(PH0_k[k], c_k);
            }
            PH0_k[k] = logistic(PH0_k[k]);
            Grad0_k[k] = filter(data, PH0_k[k], dataEdgeLength, dataEdgeLength - offset);
            H0_k[k] = bernoulli(PH0_k[k]);
        }

        float[] V1 = getVisible(H0_k, data, dataEdgeLength);

        float[][] PH1_k = new float[K][];
        float[][] Grad1_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH1_k[k] = filter(V1, W[k], dataEdgeLength, filterEdgeLength);
            if(k != 0) {
                addBias(PH0_k[k], c_k);
            }
            PH1_k[k] = logistic(PH1_k[k]);
            Grad1_k[k] = filter(V1, PH1_k[k], dataEdgeLength, dataEdgeLength - offset);
            W[k] = CD(W[k], Grad0_k[k], Grad1_k[k], learningRate);
        }

        FloatMatrix V0M = new FloatMatrix(data);
        FloatMatrix V1M = new FloatMatrix(V1);

        return (float)Math.sqrt(MatrixFunctions.pow(V0M.sub(V1M), 2.0f).sum());
    }

    private float[] filter(float[] data, float[] filter, int dataEdgeLength, int filterEdgeLength) {
        int offset = filterEdgeLength - 1;
        final int rEdgeLength = dataEdgeLength - offset;

        float[] r = new float[rEdgeLength * rEdgeLength];

        for (int y = 0; y < rEdgeLength; y++) {
            for (int x = 0; x < rEdgeLength; x++) {

                float sum = 0;
                for (int yh = 0; yh < filterEdgeLength; yh++) {
                    for (int xh = 0; xh < filterEdgeLength; xh++) {
                        int pos = (y + yh) * dataEdgeLength + x + xh;
                        sum += data[pos] * filter[yh * filterEdgeLength + xh];
                    }
                }

                r[y * rEdgeLength + x] = sum;
            }
        }

        return r;
    }

    private float[] filter(float[][] data, float[] filter, int dataEdgeLength, int filterEdgeLength, boolean normalize) {
        int offset = filterEdgeLength - 1;
        final int rEdgeLength = dataEdgeLength - offset;

        float[] r = new float[rEdgeLength * rEdgeLength];

        for (int y = 0; y < rEdgeLength; y++) {
            for (int x = 0; x < rEdgeLength; x++) {

                float sum = 0;
                for (int yh = 0; yh < filterEdgeLength; yh++) {
                    for (int xh = 0; xh < filterEdgeLength; xh++) {
                        int pos = (y + yh) * dataEdgeLength + x + xh;
                        int fPos = yh * filterEdgeLength + xh;
                        for (float[] dataK : data) {
                            sum += dataK[pos] * filter[fPos];
                        }
                    }
                }
                if(normalize) {
                    sum /= (float)data.length;
                }

                r[y * rEdgeLength + x] = sum;
            }
        }
        return r;
    }

    private float[] logistic(float[] data) {
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = 1.f / (float) (1. + Math.exp(-data[i]));
        }

        return result;
    }

    private float[] bernoulli(float[] data) {
        float[] r = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            r[i] = (data[i] > RANDOM.nextDouble()) ? 1 : 0;
        }

        return r;
    }

    private float[][] flip(float[][] h) {
        float[][] result = new float[h.length][];

        for (int i = 0; i < h.length; i++) {
            result[i] = flip(h[i]);
        }

        return result;
    }

    private float[] flip(float[] h) {
        float[] result = new float[h.length];

        for (int i = 0; i < h.length; i++) {
            result[i] = h[h.length - i - 1];
        }

        return result;
    }

    private float[] add(float[] data1, float[] data2) {
        float[] result = new float[data1.length];
        
        for (int i = 0; i < result.length; i++) {
            result[i] = data1[i] + data2[i];
        }
        
        return result;
    }

    private float[] concat(float[][] V0, float[] V1m, int dataEdgeLength, int filterEdgeLength) {
        float[] result = new float[V0[0].length];

        int offset = filterEdgeLength - 1;


        for (int y = 0; y < dataEdgeLength; y++) {
            for (int x = 0; x < dataEdgeLength; x++) {
                int pos = y * dataEdgeLength + x;
                if (y < offset || x < offset || y >= dataEdgeLength - offset || x >= dataEdgeLength - offset) {
                    for(int k = 0; k < V0.length; k++) {
                        result[pos] += V0[k][pos];
                    }
                    result[pos] /= (float)V0.length;
                } else {
                    int posm = (y - offset) * (dataEdgeLength - 2 * offset) + x - offset;
                    result[pos] = V1m[posm];
                }
            }
        }

        return result;
    }

    private float[] concat(float[] V0, float[] V1m, int dataEdgeLength, int filterEdgeLength) {
        float[] result = new float[V0.length];

        int offset = filterEdgeLength - 1;

        for (int y = 0; y < dataEdgeLength; y++) {
            for (int x = 0; x < dataEdgeLength; x++) {
                int pos = y * dataEdgeLength + x;
                if (y < offset || x < offset || y >= dataEdgeLength - offset || x >= dataEdgeLength - offset) {
                    result[pos] = V0[pos];
                } else {
                    int posm = (y - offset) * (dataEdgeLength - 2 * offset) + x - offset;
                    result[pos] = V1m[posm];
                }
            }
        }

        return result;
    }

    private float[] CD(float[] W_k, float[] Grad0_k, float[] Grad1_k, float learningRate) {
        float[] result = new float[W_k.length];
        
        for (int i = 0; i < W_k.length; i++) {
            result[i] += W_k[i] +  learningRate * (Grad0_k[i] - Grad1_k[i]);
        }
        
        return result;
    }

    public float[][][] getHidden(float[][][] data, int dataEdgeLength) {
        float[][][] result = new float[data.length][][];
        for (int i = 0; i < data.length; i++) {
            result[i] = getHidden2D(data[i], dataEdgeLength);
        }
        return result;
    }

    public float[][] getHidden2D(float[][] data, int dataEdgeLength) {
        int offset = filterEdgeLength - 1;
        int K = W.length;
        float[][] W1;

        float[][] PH0_k = new float[K][];
        float[][] Grad0_k = new float[K][];
        float[][] H0_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH0_k[k] = filter(data, W[k], dataEdgeLength, filterEdgeLength, true);
            if(k != 0) {
                addBias(PH0_k[k], c_k);
            }
            PH0_k[k] = logistic(PH0_k[k]);
            Grad0_k[k] = filter(data, PH0_k[k], dataEdgeLength, dataEdgeLength - offset, true);
            H0_k[k] = bernoulli(PH0_k[k]);
        }

        return PH0_k;
    }
    
    public float[][][] getHidden(float[][] data, int dataEdgeLength) {
        float[][][] result = new float[data.length][][];
        for (int i = 0; i < data.length; i++) {
            result[i] = getHidden(data[i], dataEdgeLength);
        }
        return result;
    }

    public float[][] getHidden(float[] data, int dataEdgeLength) {

        float[][] PH0_k = new float[W.length][];
        float[][] H0_k = new float[W.length][];

        for (int k = 0; k < W.length; k++) {
            PH0_k[k] = filter(data, W[k], dataEdgeLength, filterEdgeLength);
            if(k != 0) {
                addBias(PH0_k[k], c_k);
            }
            PH0_k[k] = logistic(PH0_k[k]);
            H0_k[k] = bernoulli(PH0_k[k]);
        }
        return PH0_k;
    }

    public float[][][] getVisible2D(float[][][] data, float[][] original, int dataEdgeLength) {
        float[][][] result = new float[data.length][data[0].length][];

        for(int i = 0; i < data.length; i++) {
            float[][] W1 = flip(W);
            int offset = filterEdgeLength - 1;
            for (int k = 0; k < W1.length; k++) {
                result[i][k] = logistic(filter(data[i], W1[k], dataEdgeLength - offset, this.filterEdgeLength, true));
//                V1m = add(V1m, r);
            }


//
//            addBias(V1m, b);
//           float[] V1 = logistic(V1m);

//            if(data != null) {
//                V1 = concat(data[i], V1, dataEdgeLength, filterEdgeLength);
//            }

        }

        return result;

    }

    public float[][] getVisible(float[][][] data, float[] original, int dataEdgeLength) {
        float[][] result = new float[data.length][];

        for(int i = 0; i < data.length; i++) {
            result[i] = getVisible(data[i], original, dataEdgeLength);
        }

        return result;
    }

    public float[] getVisible(float[][] data, float[] original, int dataEdgeLength) {

        float[][] W1 = flip(W);
        int offset = filterEdgeLength - 1;

        float[] V1m = new float[(dataEdgeLength - offset * 2) * (dataEdgeLength - offset * 2)];
        for (int k = 0; k < W1.length; k++) {
            float[] r = filter(data[k], W1[k], dataEdgeLength - offset, this.filterEdgeLength);
            V1m = add(V1m, r);
        }

        addBias(V1m, b);
        float[] V1 = logistic(V1m);
        if(original != null) {
            V1 = concat(original, V1, dataEdgeLength, filterEdgeLength);
        }

        //
        return V1;
    }

    private void addBias(float[] r, float bias) {
        for(int i = 0; i < r.length; i++) {
            r[i] += bias;
        }
    }

//    private void exportAsImage(float[][] data, String name, int count) {
//        for (int k = 0; k < data.length; k++) {
//            exportAsImage(data[k], name, count, k);
//        }
//    }
    
//    private void exportAsImage(float[] data, String name, int count){
//        exportAsImage(data, name, count, 0);
//    }
    
//    private void exportAsImage(float[] data, String name, int count, int k) {
//        new File(EXPORT_PATH + "/" + name + "/").mkdirs();
//
//        BufferedImage image = DataConverter.pixelDataToImage(data, 0.0f, false);
//        File outputfile = new File(EXPORT_PATH + "/" + name + "/" + count + "_" + k + ".png");
//        try {
//            ImageIO.write(image, "png", outputfile);
//        } catch (IOException ex) {
//            Logger.getLogger(CRBM.class.getName()).log(Level.SEVERE, null, ex);
//        }
//    }

//    private void print(float[][] W_k) {
//
//        for (int k = 0; k < K; k++) {
//            System.out.println("K:" + k);
//            for (int i = 0; i < FILTEREDGELENGTH; i++) {
//                for (int j = 0; j < FILTEREDGELENGTH; j++) {
//                    System.out.print(W_k[k][i * FILTEREDGELENGTH + j] + " ");
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }
//
//    }

    private void exportAsImage(float[][] data, String name) {

        String exportPath = "export";

        try {
            FileUtils.deleteDirectory(new File(exportPath));
        } catch (IOException e) {
            e.printStackTrace();
        }

        for(int i = 0; i < data.length; i++) {
            exportAsImage(data[i], name, i);
        }
    }

    private void exportAsImage(float[] data, String name, int count) {
        String exportPath = "export";

        new File(exportPath + "/" + name + "/").mkdirs();

        BufferedImage image = DataConverter.pixelDataToImage(data, 0.0f, false);
        File outputfile = new File(exportPath + "/" + name + "/" + count + ".png");
        try {
            ImageIO.write(image, "png", outputfile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

}
