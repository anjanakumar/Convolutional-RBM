package crbm;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.apache.commons.io.FileUtils;

/**
 * Created by Radek on 08.02.14.
 */
public class Main {

    private static final String importPath = "Data/MNIST_Small";

    private static final String exportPath = "export";
    private static final int edgeLength = 28;
    private static final int padding = 2;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = true;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;

    public static void main(String arg[]) {
        deleteOldExportData();
        Trainer trainer = new Trainer();
        trainer.train();
    }

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

    public static float checkClusters(List<Cluster> clusters, DataSet[] data) {
        System.out.println("Check clusters");
        int wrongDecision = 0;
        for (DataSet ds : data) {
            float[] d = ds.getData();
            float bestClusterDistance = Float.MAX_VALUE;
            String bestClusterLabel = "foobar";
            for (Cluster c : clusters) {
                float clusterDistance = c.distanceToCenter(d);
                if (clusterDistance < bestClusterDistance) {
                    bestClusterDistance = clusterDistance;
                    bestClusterLabel = c.getLabel();
                }
            }
            String realLabel = ds.getLabel();
            if (!(bestClusterLabel.equals(realLabel))) {
                ++wrongDecision;
                System.out.println("Found " + bestClusterLabel + " instead of " + realLabel);
            }
        }
        float error = (float) wrongDecision / data.length;
        System.out.println("Wrong: " + wrongDecision + " / " + data.length + ", Error: " + error);

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
