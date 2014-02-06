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
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author Radek
 */
public class CRBM {
    
    private final Random    RANDOM = new Random();
    
    private final String    PATH = "Data/MNIST_Small";
    
    private final int       EDGELENGTH    = 28;
    private final boolean   ISRGB       = false;
    private final boolean   BINARIZE      = true;
    private final boolean   INVERT      = true;
    private final float     MINDATA     = 0.0f;
    private final float     MAXDATA     = 1.0f;
    
    private final int       K           = 100;
    private final int       FILTEREDGELENGTH = 3;
    
    private final float     INITIALWEIGHTSSCALAR = 0.01f;
    private final float     LEARNINGRATE = 0.1f;
    
    

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new CRBM();
    }
    
    public CRBM() {
        float[][] data = loadData();
        train(data, 10);
    }
    
    private void train(float[][] data, int epochs) {
        
        int offset = FILTEREDGELENGTH - 1;
        int filterDimensions = FILTEREDGELENGTH*FILTEREDGELENGTH;
        
        float[][] W_k = new float[K][filterDimensions];
        float[][] W_kFlipped = new float[K][filterDimensions];
        
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < filterDimensions; i++) {
                W_k[k][i] = (float)(INITIALWEIGHTSSCALAR * RANDOM.nextGaussian());
            }
        }
        
        float[][] PH0_k = new float[K][];
        float[][] Grad0_k = new float[K][];
        float[][] H0_k = new float[K][];
        
        for(int e = 0; e < epochs; e++) {
            
            for(int i = 0; i < data.length; i++) {
                float[] V0 = data[i];
                
                for(int k = 0; k < K; k++) {
                    PH0_k[k] = logistic(filter(V0, W_k[k]));
                    Grad0_k[k] = filter(V0, PH0_k[k], EDGELENGTH, EDGELENGTH-offset);
                    H0_k[k] = bernoulli(PH0_k[k]);
                }

                W_kFlipped = flip(W_k);

                float[] V1m = new float[(EDGELENGTH-offset*2)*(EDGELENGTH-offset*2)];
                for(int k = 0; k < K; k++) {
                    float[] r = filter(H0_k[k], W_kFlipped[k], EDGELENGTH-offset, FILTEREDGELENGTH);
                    add(V1m, r);
                }

                float[] V1 = concat(V0, logistic(V1m));

                float[][] PH1_k = new float[K][];
                float[][] Grad1_k = new float[K][];

                for(int k = 0; k < K; k++) {
                    PH1_k[k] = logistic(filter(V1, W_k[k]));
                    Grad1_k[k] = filter(V1, PH1_k[k], EDGELENGTH, EDGELENGTH-offset);
                    CD(W_k[k], Grad0_k[k], Grad1_k[k]);
                }
            }
            
        }
        
        exportAsImage(PH0_k, "PH0_k");
        exportAsImage(H0_k, "H0_k");
    }
     private float[] filter(float[] I, float[] H) {
         return filter(I, H, EDGELENGTH, FILTEREDGELENGTH);
     }
    
    
    private float[] filter(float[] I, float[] H, int iEdgeLength, int hEdgeLength) {
        int offset = hEdgeLength - 1;
        final int rEdgeLength = iEdgeLength-offset;
        
        float[] r = new float[rEdgeLength*rEdgeLength];
                
        for(int y = 0; y < rEdgeLength; y++) {
            for(int x = 0; x < rEdgeLength; x++) {

                float sum = 0;
                for(int yh = 0; yh < hEdgeLength; yh++) {
                    for(int xh = 0; xh < hEdgeLength; xh++) {
                        int pos = (y + yh) * iEdgeLength + x + xh;
                        sum += I[pos] * H[yh * hEdgeLength + xh];
                    }
                }

                int dest = y * rEdgeLength + x;
                r[dest] = sum;
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
        
        for(int i = 0; i < imageFiles.length; i++) {
    	
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
        
        for(int v = 0; v < data.length; v++) {
            result[v] = 1.f / (float)( 1. + Math.exp(-data[v]) );
        }
        
        return result;
    }

    private float[] bernoulli(float[] data) {
        float[] r = new float[data.length];

        for(int i = 0; i < data.length; i++) {
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
        
        for(int i = 0; i < h.length; i++) {
           r[i] = h[h.length - i - 1];
        }
        
        return r;
    }

    private void add(float[] result, float[] r) {
        for(int i = 0; i < result.length; i++) {
            result[i] += r[i];
        }
    }

    private float[] concat(float[] V0, float[] V1m) {
        float[] result = new float[V0.length];
        
        int offset = FILTEREDGELENGTH-1;
        
        for (int y = 0; y < EDGELENGTH; y++) {
            for (int x = 0; x < EDGELENGTH; x++) {
                int pos = y * EDGELENGTH + x;
                if(y < offset || x < offset || y >= EDGELENGTH-offset || x >= EDGELENGTH - offset) {
                    result[pos] = V0[pos];
                } else {
                    int posm = (y-offset) * (EDGELENGTH-2*offset) + x-offset;
                    result[pos] = V1m[posm];
                }
            }
        }
        
        return result;
    }

    private void CD(float[] W_k, float[] Grad0_k, float[] Grad1_k) {
        
        for(int i = 0; i < Grad0_k.length; i++) {
            W_k[i] += LEARNINGRATE * (Grad0_k[i] - Grad1_k[i]);
        }
        
    }

    private void exportAsImage(float[][] data, String name) {
        for(int k = 0; k < data.length; k++) {
            BufferedImage image = DataConverter.pixelDataToImage(data[k], 0.0f, false);
            new File("Data/Export/" + name).mkdirs();
            File outputfile = new File("Data/Export/" + name + "/" + k + ".jpg");
            try {
                ImageIO.write(image, "jpg", outputfile);
            } catch (IOException ex) {
                Logger.getLogger(CRBM.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

}
