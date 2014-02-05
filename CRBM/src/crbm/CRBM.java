/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 *//*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package crbm;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Random;
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
    
    private final int       K           = 10;
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
        train(data);
        
    }
    
    private void train(float[][] data) {
        
        int offset = FILTEREDGELENGTH - 1;
        int filterDimensions = FILTEREDGELENGTH*FILTEREDGELENGTH;
        
        float[][] V = data;
        float[][] H_k = new float[K][(EDGELENGTH-offset)*(EDGELENGTH-offset)];
        float[][] W_k = new float[K][filterDimensions];
        
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < filterDimensions; i++) {
                W_k[k][i] = (float)(INITIALWEIGHTSSCALAR * RANDOM.nextGaussian());
            }
        }

        for(int i = 0; i < data.length; i++) {
            for(int k = 0; k < K; k++) {
                
                for(int y = 1; y < EDGELENGTH-offset; y++) {
                    for(int x = 1; x < EDGELENGTH-offset; x++) {
                        int sum = 0;
                        for(int yk = 0; yk < FILTEREDGELENGTH; yk++) {
                            for(int xk = 0; xk < FILTEREDGELENGTH; xk++) {
                                    int pos = (y + yk) * EDGELENGTH + x + xk;
                                    sum += V[i][pos] * W_k[i][yk * FILTEREDGELENGTH + xk];
                            } 
                        }

                        int dest = y * EDGELENGTH + x;
                        H_k[k][dest] = sum;
                    }
                }
                
                
                
            }
        }
        
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
    
    
}
