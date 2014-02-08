/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package crbm;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.FileUtils;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

/**
 *
 * @author Radek
 */
public class CRBM {

    private final Random RANDOM = new Random();

    private final float[][] W;
    private final int filterEdgeLength;


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

    public void train(float[][] data, int dataEdgeLength, int epochs, float learningRate, String exportPath) {

        for (int e = 0; e < epochs; e++) {
            float error = 0;
            for (float[] aData : data) {
                error += train(aData, dataEdgeLength, learningRate);
            }
            error /= data.length;
            System.out.println(error);
        }

        System.out.println("Training finished");
    }

    private float train(float[] data, int dataEdgeLength, float learningRate) {
        int offset = filterEdgeLength - 1;
        int K = W.length;
        float[][] W1;

        float[][] PH0_k = new float[K][];
        float[][] Grad0_k = new float[K][];
        float[][] H0_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH0_k[k] = logistic(filter(data, W[k],dataEdgeLength, filterEdgeLength));
            Grad0_k[k] = filter(data, PH0_k[k], dataEdgeLength, dataEdgeLength - offset);
            H0_k[k] = bernoulli(PH0_k[k]);
        }

        //H0_k = maxPooling(H0_k, 2);

        W1 = flip(W);

        float[] V1m = new float[(dataEdgeLength - offset * 2) * (dataEdgeLength - offset * 2)];
        for (int k = 0; k < K; k++) {
            float[] r = filter(PH0_k[k], W1[k], dataEdgeLength - offset, this.filterEdgeLength);
            V1m = add(V1m, r);
        }

        float[] V1 = concat(data, logistic(V1m), dataEdgeLength, filterEdgeLength);

        float[][] PH1_k = new float[K][];
        float[][] Grad1_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH1_k[k] = logistic(filter(V1, W[k],dataEdgeLength, filterEdgeLength));
            Grad1_k[k] = filter(V1, PH1_k[k], dataEdgeLength, dataEdgeLength - offset);
            W[k] = CD(W[k], Grad0_k[k], Grad1_k[k], learningRate);
        }
        
        FloatMatrix V0M = new FloatMatrix(data);
        FloatMatrix V1M = new FloatMatrix(V1);

        return (float)Math.sqrt(MatrixFunctions.pow(V0M.sub(V1M), 2.0f).sum());

//        if(print) {
//            print(W);
//            exportAsImage(PH0_k, "PH0_k", count);
//            exportAsImage(PH1_k, "PH1_k", count);
//            exportAsImage(H0_k, "H0_k", count);
//            exportAsImage(data, "data", count);
//            exportAsImage(V1, "recon", count);
//        }
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
            PH0_k[k] = logistic(filter(data, W[k], dataEdgeLength, filterEdgeLength));
            H0_k[k] = bernoulli(PH0_k[k]);
        }
        return H0_k;
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

}
