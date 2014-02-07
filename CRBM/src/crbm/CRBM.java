/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package crbm;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.apache.commons.io.FileUtils;

/**
 *
 * @author Radek
 */
public class CRBM {

    private final Random RANDOM = new Random();

    private final String FILEPATH = "export";
    private final String PATH = "Data/MNIST_Small";

    private final int EDGELENGTH = 28;
    private final boolean ISRGB = false;
    private final boolean BINARIZE = true;
    private final boolean INVERT = true;
    private final float MINDATA = 0.0f;
    private final float MAXDATA = 1.0f;

    private final int EPOCHS = 10000;
    private final int K = 3;
    private final int FILTEREDGELENGTH = 3;

    private final float INITIALWEIGHTSSCALAR = 1f;
    private final float LEARNINGRATE = 0.01f;
    
    private final int POOLING_SIZE = 2;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new CRBM();
    }

    public CRBM() {
        float[][] data = loadData();

        train(data);
    }

    private void train(float[][] data) {
        try {
            FileUtils.deleteDirectory(new File("/Users/Radek/export"));
        } catch (IOException ex) {
            Logger.getLogger(CRBM.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        int filterDimensions = FILTEREDGELENGTH * FILTEREDGELENGTH;

        float[][] W_k = new float[K][filterDimensions];

        for (int k = 0; k < K; k++) {
            for (int i = 0; i < filterDimensions; i++) {
                W_k[k][i] = (float) (INITIALWEIGHTSSCALAR * ((RANDOM.nextDouble() - 0.5) * 2) / (FILTEREDGELENGTH * FILTEREDGELENGTH));
            }
        }

        boolean print = false;
        for (int e = 0; e < EPOCHS; e++) {
            if(e == EPOCHS - 1) print = true;
            for (int i = 0; i < data.length; i++) {
                trainImage(data[i], W_k, i, print);
            }
        }
    }

    private void trainImage(float[] data, float[][] W_k, int count, boolean print) {
        int offset = FILTEREDGELENGTH - 1;
        int filterDimensions = FILTEREDGELENGTH * FILTEREDGELENGTH;

        float[][] W_kFlipped = new float[K][filterDimensions];

        float[][] PH0_k = new float[K][];
        float[][] Grad0_k = new float[K][];
        float[][] H0_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH0_k[k] = logistic(filter(data, W_k[k]));
            Grad0_k[k] = filter(data, PH0_k[k], EDGELENGTH, EDGELENGTH - offset);
            H0_k[k] = bernoulli(PH0_k[k]);
        }
        
        /*
        float[][] max = maxPooling(H0_k, POOLING_SIZE);
        for (int k = 0; k < K; k++) {
            H0_k[k] = mul(H0_k[k], max[k]);
        }
        */

        W_kFlipped = flip(W_k);

        float[] V1m = new float[(EDGELENGTH - offset * 2) * (EDGELENGTH - offset * 2)];
        for (int k = 0; k < K; k++) {
            float[] r = filter(H0_k[k], W_kFlipped[k], EDGELENGTH - offset, FILTEREDGELENGTH);
            add(V1m, r);
        }

        float[] V1 = concat(data, logistic(V1m));

        float[][] PH1_k = new float[K][];
        float[][] Grad1_k = new float[K][];

        for (int k = 0; k < K; k++) {
            PH1_k[k] = logistic(filter(V1, W_k[k]));
            Grad1_k[k] = filter(V1, PH1_k[k], EDGELENGTH, EDGELENGTH - offset);
            CD(W_k[k], Grad0_k[k], Grad1_k[k]);
        }

        print(W_k);
        if(print) {
            exportAsImage(PH0_k, "PH0_k", count);
            exportAsImage(PH1_k, "PH1_k", count);
        }
    }

    private float[] filter(float[] I, float[] H) {
        return filter(I, H, EDGELENGTH, FILTEREDGELENGTH);
    }

    private float[] filter(float[] I, float[] H, int iEdgeLength, int hEdgeLength) {
        int offset = hEdgeLength - 1;
        final int rEdgeLength = iEdgeLength - offset;

        float[] r = new float[rEdgeLength * rEdgeLength];

        float sumH = 0f;
        for (int i = 0; i < H.length; i++) {
            sumH += Math.abs(H[i]);
        }

        for (int y = 0; y < rEdgeLength; y++) {
            for (int x = 0; x < rEdgeLength; x++) {

                float sum = 0;
                float sumA = 0.0f;
                for (int yh = 0; yh < hEdgeLength; yh++) {
                    for (int xh = 0; xh < hEdgeLength; xh++) {
                        int pos = (y + yh) * iEdgeLength + x + xh;
                        sum += I[pos] * H[yh * hEdgeLength + xh];
                    }
                }

                int dest = y * rEdgeLength + x;
                r[dest] = (sum / sumH);
            }
        }

        return r;
    }

    public float[][] loadData() {

        File imageFolder = new File(PATH);
        final File[] imageFiles = imageFolder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return (name.endsWith("jpg") || name.endsWith("png") || name.endsWith("gif"));
            }
        });

        int size = EDGELENGTH * EDGELENGTH;
        float[][] data = new float[imageFiles.length][size];

        for (int i = 0; i < imageFiles.length; i++) {

            float[] imageData;
            try {
                imageData = DataConverter.processPixelData(ImageIO.read(imageFiles[i]), EDGELENGTH, BINARIZE, INVERT, MINDATA, MAXDATA, ISRGB);
            } catch (IOException e) {
                System.out.println("Could not load: " + imageFiles[i].getAbsolutePath());
                return null;
            }

            data[i] = imageData;

        }

        return data;
    }

    private float[] logistic(float[] data) {
        float[] result = new float[data.length];

        for (int v = 0; v < data.length; v++) {
            result[v] = 1.f / (float) (1. + Math.exp(-data[v]));
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
        float[][] r = new float[h.length][];

        for (int i = 0; i < r.length; i++) {
            r[i] = flip(h[i]);
        }

        return r;
    }

    private float[] flip(float[] h) {
        float[] r = new float[h.length];

        for (int i = 0; i < h.length; i++) {
            r[i] = h[h.length - i - 1];
        }

        return r;
    }

    private void add(float[] result, float[] r) {
        for (int i = 0; i < result.length; i++) {
            result[i] += r[i];
        }
    }

    private float[] concat(float[] V0, float[] V1m) {
        float[] result = new float[V0.length];

        int offset = FILTEREDGELENGTH - 1;

        for (int y = 0; y < EDGELENGTH; y++) {
            for (int x = 0; x < EDGELENGTH; x++) {
                int pos = y * EDGELENGTH + x;
                if (y < offset || x < offset || y >= EDGELENGTH - offset || x >= EDGELENGTH - offset) {
                    result[pos] = V0[pos];
                } else {
                    int posm = (y - offset) * (EDGELENGTH - 2 * offset) + x - offset;
                    result[pos] = V1m[posm];
                }
            }
        }

        return result;
    }

    private void CD(float[] W_k, float[] Grad0_k, float[] Grad1_k) {

        for (int i = 0; i < W_k.length; i++) {
            W_k[i] += LEARNINGRATE * (Grad0_k[i] - Grad1_k[i]);
        }

    }

    private void exportAsImage(float[][] data, String name, int count) {
        try {            
            new File(FILEPATH + "/" + name + "/").mkdirs();
            
            for (int k = 0; k < data.length; k++) {
                BufferedImage image = DataConverter.pixelDataToImage(data[k], 0.0f, false);
                File outputfile = new File(FILEPATH + "/" + name + "/" + count + "_" + k + ".png");

                ImageIO.write(image, "png", outputfile);
            }
        } catch (IOException ex) {
            Logger.getLogger(CRBM.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void print(float[][] W_k) {

        for (int k = 0; k < K; k++) {
            System.out.println("K:" + k);
            for (int i = 0; i < FILTEREDGELENGTH; i++) {
                for (int j = 0; j < FILTEREDGELENGTH; j++) {
                    System.out.print(W_k[k][i * FILTEREDGELENGTH + j] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }

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

    private float[][] maxPooling(float[][] data, int poolingSize) {
        int offset = FILTEREDGELENGTH-1;
        
        int pEdgeLength = EDGELENGTH - offset;
        int iEdgeLength = EDGELENGTH - offset;
        
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

                            Math.max(value, max);
                            
                        }
                    }

                    result[k][y * rEdgeLength + x] = max;
                }
            }
        }
        
        
        for (int y = 0; y < rEdgeLength; y++) {
            for (int x = 0; x < rEdgeLength; x++) {
                int pos = y * rEdgeLength +x;
                
                boolean found = false;
                boolean foundSecond = false;
                int iK = 0;
                
                for(int k = 0; k < K; k++) {
                    if(result[k][pos] > 0.5f) {
                        if(found) {
                            result[k][pos] = 0.0f;
                            foundSecond = true;
                        } else {
                            iK = k;
                            found = true;
                        }
                    }
                }
                if(foundSecond){
                    result[iK][pos] = 0.0f;
                }
                
            }
        }
        
        float[][] superResult = new float[K][];
        for(int k = 0; k < K; k++) {
            superResult[k] = nearestNeighbour(result[k], rEdgeLength, rEdgeLength, iEdgeLength, iEdgeLength);
        }

        return superResult;
    }

    private float[] mul(float[] m1, float[] m2) {
        float[] result = new float[m1.length];
        
        for(int i = 0; i < m1.length; i++) {
            result[i] = m1[i] * m2[i];
        }
        
        return result;
    }

}
