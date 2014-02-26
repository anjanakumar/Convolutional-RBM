package crbm.rbm;

import crbm.*;
import org.apache.commons.io.FileUtils;
import org.jblas.FloatMatrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.*;

/**
 * Created by Moritz on 2/25/14.
 */
public class SRBMTest {
    private static final boolean exportImages = true;
    private static final String exportPath = "export";

    private static final int edgeLength = 28;
    private static final int padding = 0;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = true;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;

    // data for training
    private static final String trainingDataPath = "CRBM/Data/MNIST_5";
    // data for testing the generated clusters
    private static final String testDataPath = "CRBM/Data/MNIST_1000_Test_Database";

    public static void main(String[] args) {
        deleteOldExportData();
        DataSet[] trainingDataSet = loadData(trainingDataPath);
        float[][] trainingData = dataSetToArray(trainingDataSet);

        SRBM srbm = new SRBM(edgeLength * edgeLength, 0.01f, 10, new DefaultLogisticMatrixFunction());
        srbm.train(trainingData, new StoppingCondition(5000),false, false);

        float[][] hidden = srbm.getHidden(trainingData, false);
        float[][] visible = srbm.getVisible(hidden, false);

        float[][] features = FloatMatrix.eye(srbm.getWeights()[0].length).toArray2();
        float[][] featurePics = srbm.getVisible(features, false);

        exportAsImage(trainingData, "original", 0);
        exportAsImage(visible, "recon", 0);
        exportAsImage(featurePics, "features", 0);
    }

    /**
     * loads the image data from a directory and converts into a data structure
     * @param importPath
     * @return
     */
    public static DataSet[] loadData(String importPath) {

        File imageFolder = new File(importPath);
        final File[] imageFiles = imageFolder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return (name.endsWith("jpg") || name.endsWith("png") || name.endsWith("gif"));
            }
        });

        int size = edgeLength * edgeLength;
        DataSet[] result = new DataSet[imageFiles.length];

        for (int i = 0; i < imageFiles.length; i++) {
            float[] imageData;
            try {
                imageData = DataConverter.processPixelData(ImageIO.read(imageFiles[i]), edgeLength, binarize, invert, minData, maxData, isRGB);
            } catch (IOException e) {
                System.out.println("Could not load: " + imageFiles[i].getAbsolutePath());
                return null;
            }

            imageData = pad(imageData, edgeLength, padding);

            String label = imageFiles[i].getName().split("_")[0];
            result[i] = new DataSet(imageData, label);

        }

        return result;
    }

    public static float[][] dataSetToArray(DataSet[] dataSet) {
        float[][] result = new float[dataSet.length][];
        for (int i = 0; i < dataSet.length; ++i) {
            result[i] = dataSet[i].getData();
        }
        return result;
    }

    /**
     * adds a padding to the training images, so that the filtered image
     * has the original size
     * @param data
     * @param dataEdgeLength
     * @param padding
     * @return
     */
    private static float[] pad(float[] data, int dataEdgeLength, int padding) {
        int newEdgeLength = dataEdgeLength + padding * 2;
        float[] result = new float[newEdgeLength * newEdgeLength];

        for (int y = 0; y < newEdgeLength; y++) {
            for (int x = 0; x < newEdgeLength; x++) {

                int pos = y * newEdgeLength + x;
                if (y < padding || x < padding || y >= dataEdgeLength + padding || x >= dataEdgeLength + padding) {
                    result[pos] = 0.0f;
                } else {
                    int posm = (y - padding) * (newEdgeLength - padding * 2) + x - padding;
                    result[pos] = data[posm];
                }

            }
        }

        return result;
    }

    public static DataSet[] arrayToDataSet(float[][] resultData, DataSet[] originalData) {
        //Length of result data must be equal to length of original data, eg. number of pics
        if (resultData.length != originalData.length) {
            return null;
        }
        DataSet[] result = new DataSet[resultData.length];
        for (int i = 0; i < resultData.length; ++i) {
            result[i] = new DataSet(resultData[i], originalData[i].getLabel());
        }
        return result;
    }

    /**
     * Generates clusters from the hidden layer result of the last rbm
     * It uses training data labels to put all data from one category into one cluster
     * The cluster center is the mean value of the data in the cluster
     * @param data
     * @return
     */
    public static List<Cluster> generateClusters(DataSet[] data) {
        List<Cluster> clusters = new LinkedList<Cluster>();

        for (DataSet ds : data) {
            boolean found = false;
            String label = ds.getLabel();
            for (Cluster c : clusters) {
                if (c.getLabel().equals(label)) {
                    c.addVector(ds.getData());
                    found = true;
                    break;
                }
            }
            if (!found) {
                Cluster c = new Cluster(label);
                c.addVector(ds.getData());
                clusters.add(c);
            }
        }
        for (Cluster c : clusters) {
            c.init();
        }

        return clusters;
    }

    /**
     * Checks if the test data is set to the right cluster
     * e.g apples should be set to the apple labeled cluster
     * @param clusters
     * @param data
     * @return
     */
    public static float checkClusters(List<Cluster> clusters, DataSet[] data) {
        System.out.println("Check clusters");
        int numberOfClusters = clusters.size();

        // Init ConfusionMatrix
        ArrayList<String> labels = new ArrayList<String>();
        for (Cluster c : clusters) {
            labels.add(c.getLabel());;
        }
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(labels);

        float rank = 0;
        int wrongDecision = 0;
        for (DataSet ds : data) {
            HashMap<String, Float> clusterDistances = new HashMap<String, Float>();

            float[] d = ds.getData();

            for (Cluster c : clusters) {
                String label = c.getLabel();
                float clusterDistance = c.distanceToCenter(d);
                clusterDistances.put(label, clusterDistance);
            }

            ValueComparator valueComp = new ValueComparator(clusterDistances);
            TreeMap<String, Float> sortedDistances = new TreeMap<String, Float>(valueComp);

            sortedDistances.putAll(clusterDistances);

            String realLabel = ds.getLabel();

            String bestClusterLabel = sortedDistances.firstEntry().getKey();

            confusionMatrix.addEntry(realLabel, bestClusterLabel);

            int pos = 0;
            for (String key : sortedDistances.keySet()) {
                ++pos;
                if(key.equals(realLabel)) break;
            }

            rank += 1.0f / (float)pos;
            if (!(bestClusterLabel.equals(realLabel))) {
                ++wrongDecision;
                System.out.println("Found " + bestClusterLabel + " instead of " + realLabel);
            }
        }

        float meanRank = rank / (float)data.length;
        float error = (float) wrongDecision / data.length;
        float overallCorrectRate = (float)(data.length - wrongDecision) / data.length;

        System.out.println("Mean Rank: " + meanRank);
        System.out.println("Wrong: " + wrongDecision + " / " + data.length + ", Error: " + error);
        System.out.println("Right: " + (data.length - wrongDecision) + " / " + data.length + ", OverallCorrectRate: " + overallCorrectRate);

        System.out.println("Confusion Matrix: " + meanRank);
        System.out.println(confusionMatrix.toString());

        return error;
    }

    public static void printClusters(List<Cluster> clusters) {
        System.out.println("Number of Clusters: " + clusters.size());
        int i = 0;
        for (Cluster cluster: clusters) {
            System.out.println();
            System.out.println("Cluster " + i + ": " + cluster.getLabel());
            float[] center = cluster.getCenter();
            System.out.print("Center:");

            for (float f : center) {
                System.out.print(" " + f);
            }

            System.out.println();
            i++;
        }
    }

    public static void deleteOldExportData() {
        try {
            FileUtils.deleteDirectory(new File(exportPath));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void exportAsImage(float[][][][] data, String name) {
        for (int i = 0; i < data.length; i++) {
            exportAsImage(data[i], name + " " + i);
        }
    }

    public static void exportAsImage(float[][][] data, String name) {
        for (int i = 0; i < data.length; i++) {
            exportAsImage(data[i], name, i);
        }
    }

    public static void exportAsImage(float[][] data, String name, int count) {
        for (int i = 0; i < data.length; i++) {
            exportAsImage(data[i], name, count, i);
        }
    }

    public static void exportAsImage(float[] data, String name, int count, int index) {
        if(!exportImages) return;
        new File(exportPath + "/" + name + "/").mkdirs();

        BufferedImage image = DataConverter.pixelDataToImage(data, 0.0f, false);
        File outputfile = new File(exportPath + "/" + name + "/" + count + "_" + index + ".png");
        try {
            ImageIO.write(image, "png", outputfile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}

/**
 * code from http://stackoverflow.com/a/1283722
 */
class ValueComparator implements Comparator<String> {

    Map<String, Float> base;
    public ValueComparator(Map<String, Float> base) {
        this.base = base;
    }

    // Note: this comparator imposes orderings that are inconsistent with equals.
    @Override
    public int compare(String a, String b) {
        if (base.get(a) >= base.get(b)) {
            return 1;
        } else {
            return -1;
        } // returning 0 would merge keys
    }
}
