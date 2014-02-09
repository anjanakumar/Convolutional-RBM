package crbm;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * Created by Radek on 08.02.14.
 */
public class Main {


    // DATA
    private static final String IMPORT_PATH = "CRBM/Data/MNIST_Small";
    private static final int EDGELENGTH = 28;
    private static final boolean ISRGB = false;
    private static final boolean BINARIZE = true;
    private static final boolean INVERT = true;
    private static final float MINDATA = 0.0f;
    private static final float MAXDATA = 1.0f;
    
    private static final float CLUSTER_DISTANCE_FACTOR = 5f;

    public static void main(String arg[]) {
        Trainer trainer = new Trainer();
        trainer.train();
    }

    public static float[][] loadData() {

        File imageFolder = new File(IMPORT_PATH);
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

    public static Cluster[] clustering(float[][] data){
        if(data == null || data.length == 0) return null;
        int len = data[0].length;
        
        // the maxDistance dependends on the data dimensionality
        float maxDistance = CLUSTER_DISTANCE_FACTOR * len;
                
        Random random = new Random();
        Cluster[] result = new Cluster[1];
        result[0] = new Cluster(data);
        float resultDistance = result[0].getTotalDistance();
        
        // add one new cluster in each iteration
        // until the total distance of all data 
        // to their cluster centers is small enough
        while(resultDistance > maxDistance){
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
