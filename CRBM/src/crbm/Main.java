package crbm;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Radek on 08.02.14.
 */
public class Main {
    
    //private static final String importPath = "Data/MNIST_Small";
    private static final String importPath = "Data/MNIST_Small";
    private static final int edgeLength = 28;
    private static final int padding = 2;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = true;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    
    private static final float maxClusterDistance = 20f;

    public static void main(String arg[]) {
        Trainer trainer = new Trainer();
        trainer.train();
    }

    public static float[][] loadData() {

        File imageFolder = new File(importPath);
        final File[] imageFiles = imageFolder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return (name.endsWith("jpg") || name.endsWith("png") || name.endsWith("gif"));
            }
        });

        int size = edgeLength * edgeLength;
        float[][] data = new float[imageFiles.length][size];

        for (int i = 0; i < imageFiles.length; i++) {

            float[] imageData;
            try {
                imageData = DataConverter.processPixelData(ImageIO.read(imageFiles[i]), edgeLength, binarize, invert, minData, maxData, isRGB);
            } catch (IOException e) {
                System.out.println("Could not load: " + imageFiles[i].getAbsolutePath());
                return null;
            }

            data[i] = pad(imageData, edgeLength, padding);

        }

        return data;
    }

    private static float[] pad(float[] data, int dataEdgeLength, int padding) {
        int newEdgeLength = dataEdgeLength + padding*2;
        float[] result = new float[newEdgeLength*newEdgeLength];

        for (int y = 0; y < newEdgeLength; y++) {
            for (int x = 0; x < newEdgeLength; x++) {

                int pos = y * newEdgeLength + x;
                if (y < padding || x < padding || y >= dataEdgeLength + padding || x >= dataEdgeLength + padding) {
                    result[pos] = 0.0f;
                } else {
                    int posm = (y - padding) * (newEdgeLength - padding*2) + x - padding;
                    result[pos] = data[posm];
                }

            }
        }

        return result;
    }
    
    public static Cluster[] clustering(float[][] data){
        if(data == null || data.length == 0) return null;
        int len = data[0].length;

        Random random = new Random();
        Cluster[] result = new Cluster[1];
        result[0] = new Cluster(data);
        float resultDistance = result[0].getTotalDistance();

        // add one new cluster in each iteration
        // until the total distance of all data
        // to their cluster centers is small enough
        while(resultDistance > maxClusterDistance){
            // find worst cluster
            float worstTotalDistance = 0;
            int clusterIndex = 0;
            for(int i = 0; i < result.length; ++i){
                float cd = result[i].getTotalDistance();
                if(cd > worstTotalDistance){
                    worstTotalDistance = cd;
                    clusterIndex = i;
                }
            }
            // add additional cluster slot
            Cluster[] tmp = new Cluster[result.length + 1];
            for(int i = 0; i < result.length; ++i){
                tmp[i] = new Cluster(result[i].getCenter());
            }
            // split worst cluster into two separated clusters
            float[] cOld = result[clusterIndex].getCenter();
            float[] c1 = new float[len];
            float[] c2 = new float[len];
            for(int i = 0; i < len; ++i){
                float r = (random.nextFloat() - .5f) * .2f;
                c1[i] = cOld[i] + r;
                c2[i] = cOld[i] - r;
            }
            tmp[clusterIndex] = new Cluster(c1);
            tmp[result.length] = new Cluster(c2);
            // assign data vectors to new clusters
            for(float[] v : data){
                float bestClusterDistance = Float.MAX_VALUE;
                int bestClusterIndex = -1;
                for(int i = 0; i < tmp.length; ++i){
                    float cd = tmp[i].distanceToCenter(v);
                    if(cd < bestClusterDistance){
                        bestClusterDistance = cd;
                        bestClusterIndex = i;
                    }
                }
                tmp[bestClusterIndex].addVector(v);
            }
            // initialize new clusters and calculate new distance
            resultDistance = 0;
            for(Cluster c : tmp){
                c.init();
                resultDistance += c.getTotalDistance();
            }
            result = tmp;
        }
        return result;
    }
}
