/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package crbm;

import java.util.Random;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

/**
 *
 * @author Radek
 */
public class CRBM {

    private final Random RANDOM = new Random();

    private float[][] W;
    private float[][][] W3D;
    private int K;
    private int filterEdgeLength;

    private int dataEdgeLength = 0;


    public CRBM(int K, int filterEdgeLength, int dataEdgeLength) {
        this.K = K;
        this.filterEdgeLength = filterEdgeLength;
        this.dataEdgeLength = dataEdgeLength;
        this.W = initW(K, filterEdgeLength);
    }

    private float[][] initW(int K, int filterEdgeLength) {
        int filterDimensions = filterEdgeLength * filterEdgeLength;
        float[][] result = new float[K][filterDimensions];

        for (int k = 0; k < K; k++) {
            for (int h = 0; h < filterDimensions; h++) {
                result[k][h] = (RANDOM.nextFloat() - 0.5f) * 2;
            }
        }

        return result;
    }

    public void train(float[][] data, int epochs, float learningRate) {

        System.out.println("Training started");

        for (int e = 0; e < epochs; e++) {
            float error = train(data, learningRate);
            System.out.println("Error: " + error);
        }

        System.out.println("Training finished");
    }


    public void train(float[][][] data, int epochs, float learningRate) {

        int filterDimensions = filterEdgeLength * filterEdgeLength;

        W3D = new float[K][data[0].length][filterDimensions];

        for (int k = 0; k < K; k++) {
            for(int kBefore = 0; kBefore < data[0].length; kBefore++) {
                for (int h = 0; h < filterDimensions; h++) {
                    W3D[k][kBefore][h] = (RANDOM.nextFloat() - 0.5f) * 2;
                }
            }
        }

        System.out.println("Training started");

        for (int e = 0; e < epochs; e++) {
            float error = train(data, learningRate);
            System.out.println("Error: " + error);
        }

        System.out.println("Training finished");
    }


    private float train(float[][][] V0, float learningRate) {
        float[][][][] PH0 = new float[V0.length][K][][];
        float[][][][] Grad0 = new float[V0.length][K][][];
        float[][][][] H0 = new float[V0.length][K][][];

        float[][][][] PH1 = new float[V0.length][K][][];
        float[][][][] Grad1 = new float[V0.length][K][][];

        float[][][] V1 = new float[V0.length][][];
        float[][][] V1m = new float[V0.length][][];

        float error = 0.0f;
        for (int i = 0; i < V0.length; i++) {

            for (int k = 0; k < K; k++) {
                PH0[i][k] = convolution(V0[i], W3D[k]);
                PH0[i][k] = logistic(PH0[i][k]);
                H0[i][k] = bernoulli(PH0[i][k]);
                Grad0[i][k] = convolution(V0[i], PH0[i][k]);
            }

            V1m[i] = getVisible(PH0[i]);
            V1[i] = concatenate(V0[i], V1m[i]);

            for (int k = 0; k < K; k++) {
                PH1[i][k] = convolution(V1[i], W3D[k]);
                PH1[i][k] = logistic(PH1[i][k]);
                Grad1[i][k] = convolution(V1[i], PH1[i][k]);

                for(int kBefore = 0; kBefore < W3D[0].length; kBefore++) {
                    for(int h = 0; h < W3D[0][0].length; h++) {
                        W3D[k][kBefore][h] = W3D[k][kBefore][h] + learningRate * (Grad0[i][k][kBefore][h] - Grad1[i][k][kBefore][h]);
                    }
                }
            }

            for (int k = 0; k < K; k++) {
                FloatMatrix V0M = new FloatMatrix(PH0[i][k]);
                FloatMatrix V1M = new FloatMatrix(PH1[i][k]);

                error += (float)Math.sqrt(MatrixFunctions.pow(V0M.sub(V1M), 2.0f).sum());
            }
            error /= (float)K;
        }

        return error / (float)V0.length;

    }

    private float train(float[][] V0, float learningRate) {

        float[][][] PH0 = new float[V0.length][K][];
        float[][][] Grad0 = new float[V0.length][K][];
        float[][][] H0 = new float[V0.length][K][];

        float[][][] PH1 = new float[V0.length][K][];
        float[][][] Grad1 = new float[V0.length][K][];

        float[][] V1 = new float[V0.length][];
        float[][] V1m = new float[V0.length][];

        float error = 0.0f;
        for (int i = 0; i < V0.length; i++) {

            for (int k = 0; k < K; k++) {
                PH0[i][k] = convolution(V0[i], W[k]);
                PH0[i][k] = logistic(PH0[i][k]);
                Grad0[i][k] = convolution(V0[i], PH0[i][k]);
                H0[i][k] = bernoulli(PH0[i][k]);
            }

            V1m[i] = getVisible(PH0[i]);
            V1[i] = concatenate(V0[i], V1m[i]);

            for (int k = 0; k < K; k++) {
                PH1[i][k] = convolution(V1[i], W[k]);
                PH1[i][k] = logistic(PH1[i][k]);
                Grad1[i][k] = convolution(V1[i], PH1[i][k]);

                for(int h = 0; h < W[0].length; h++) {
                    W[k][h] = W[k][h] + learningRate * (Grad0[i][k][h] - Grad1[i][k][h]);
                }
            }

            FloatMatrix V0M = new FloatMatrix(Grad0[i]);
            FloatMatrix V1M = new FloatMatrix(Grad1[i]);

            error += (float)Math.sqrt(MatrixFunctions.pow(V0M.sub(V1M), 2.0f).sum());
        }

        return error / (float)V0.length;
    }

    public float[][][][] getHidden(float[][][] data) {
        float[][][][] PH0 = new float[data.length][K][][];
        float[][][][] Grad0 = new float[data.length][K][][];

        for (int i = 0; i < data.length; i++) {
            for (int k = 0; k < W.length; k++) {
                PH0[i][k] = convolution(data[i], W3D[k]);
                PH0[i][k] = logistic(PH0[i][k]);
            }
        }

        return PH0;
    }

    public float[][][] getHidden(float[][] data) {
        float[][][] PH0 = new float[data.length][K][];
        float[][][] H0 = new float[data.length][K][];

        for (int i = 0; i < data.length; i++) {
            for (int k = 0; k < W.length; k++) {
                PH0[i][k] = convolution(data[i], W[k]);
                PH0[i][k] = logistic(PH0[i][k]);
                H0[i][k] = bernoulli(PH0[i][k]);
            }
        }

        return PH0;
    }

    public float[][][] getVisible(float[][][][] data) {
        float[][][] result = new float[data.length][][];

        for(int i = 0; i < data.length; i++) {
            result[i] = getVisible(data[i]);
        }

        return result;
    }

    public float[][] getVisible(float[][][] data) {
        float[][][] W1 = flip(W3D);

        int resultingDataEdgeLength = getDataEdgeLengthAfterConvolution(data[0][0], W1[0][0]);
        float[][] v1m = new float[data.length][resultingDataEdgeLength * resultingDataEdgeLength];

        for (int k = 0; k < data.length; k++) {
            float[][] r = convolution(data[k], W1[k]);
            add(v1m, r);
        }
        v1m = logistic(v1m);

        return v1m;
    }

    public float[] getVisible(float[][] data) {
        float[][] W1 = flip(W);

        int resultingDataEdgeLength = getDataEdgeLengthAfterConvolution(data[0], W1[0]);
        float[] v1m = new float[resultingDataEdgeLength * resultingDataEdgeLength];
        for (int k = 0; k < W1.length; k++) {
            float[] r = convolution(data[k], W1[k]);
            add(v1m, r);
        }
        v1m = logistic(v1m);

        return v1m;
    }

    private float[][] convolution(float[][] data, float[][] filter) {
        final int dataEdgeLength = getDataEdgeLength(data[0]);
        final int filterEdgeLength = getDataEdgeLength(filter[0]);
        final int resultingEdgeLength = getDataEdgeLengthAfterConvolution(data[0], filter[0]);

        float[][] result = new float[data.length][resultingEdgeLength * resultingEdgeLength];

        for (int k = 0; k < data.length; k++) {
            result[k] = convolution(data[k], filter[k]);
        }

        return result;
    }

    private float[] convolution(float[] data, float[] filter) {
        final int dataEdgeLength = getDataEdgeLength(data);
        final int filterEdgeLength = getDataEdgeLength(filter);
        final int resultingEdgeLength = getDataEdgeLengthAfterConvolution(data, filter);

        float[] result = new float[resultingEdgeLength * resultingEdgeLength];

        for (int y = 0; y < resultingEdgeLength; y++) {
            for (int x = 0; x < resultingEdgeLength; x++) {

                float sum = 0;
                for (int yh = 0; yh < filterEdgeLength; yh++) {
                    for (int xh = 0; xh < filterEdgeLength; xh++) {
                        int pos = (y + yh) * dataEdgeLength + x + xh;
                        sum += data[pos] * filter[yh * filterEdgeLength + xh];
                    }
                }

                result[y * resultingEdgeLength + x] = sum;
            }
        }

        return result;
    }

    private float[][] logistic(float[][] data) {
        float[][] result = new float[data.length][];

        for (int i = 0; i < data.length; i++) {
            result[i] = logistic(data[i]);
        }

        return result;
    }

    private float[] logistic(float[] data) {
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = 1.f / (float) (1. + Math.exp(-data[i]));
        }

        return result;
    }

    private float[][] bernoulli(float[][] data) {
        float[][] result = new float[data.length][];

        for (int i = 0; i < data.length; i++) {
            result[i] = bernoulli(data[i]);
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

    private float[][][] flip(float[][][] h) {
        float[][][] result = new float[h.length][][];

        for (int i = 0; i < h.length; i++) {
            result[i] = flip(h[i]);
        }

        return result;
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

    private float[][] concatenate(float[][] V0, float[][] V1m) {
        float[][] result = new float[V0.length][];

        for(int i = 0; i < V0.length; i++) {
            result[i] = concatenate(V0[i], V1m[i]);
        }

        return result;
    }

    private float[] concatenate(float[] V0, float[] V1m) {
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

    private void add(float[][] data1, float[][] data2) {
        for (int i = 0; i < data1.length; i++) {
            add(data1[i], data2[i]);
        }
    }

    private void add(float[] data1, float[] data2) {
        for (int i = 0; i < data1.length; i++) {
            data1[i] += data2[i];
        }
    }

    public int getDataEdgeLength(float[] data) {
        return (int)Math.sqrt(data.length);
    }

    private int getDataEdgeLengthAfterConvolution(float[] data, float[] filter) {
        return getDataEdgeLengthAfterConvolution(getDataEdgeLength(data), getDataEdgeLength(filter));
    }

    private int getDataEdgeLengthAfterConvolution(int dataEdgeLength, int filterEdgeLength) {
        return dataEdgeLength - filterEdgeLength + 1;
    }

}
