package crbm;

import org.apache.commons.io.FileUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;
import java.io.File;

/**
 * Created by Radek on 08.02.14.
 */
public class Trainer {

    private int K = 15;
    private float learningRate = 0.01f;
    private int epochs = 100;

    private int crbmFilterEdgeLength = 5;

    private int crbm1DataEdgeLength = 32;
    private int crbm1PoolingSize = 2;

    private int crbm2DataEdgeLength = crbm1DataEdgeLength - crbmFilterEdgeLength + 1;
    private int crbm2PoolingSize = 2;


    public Trainer() {

    }

    public void train() {
        String exportPath = "export";
        try {
            FileUtils.deleteDirectory(new File(exportPath));
        } catch (IOException e) {
            e.printStackTrace();
        }

        float[][] data = Main.loadData();

        exportAsImage(data, "test");

        CRBM crbm1 = new CRBM(K, crbmFilterEdgeLength);
        crbm1.train(data, crbm1DataEdgeLength, epochs, learningRate, "First-RBM");
        crbm1.killFirst();
        float[][][] hidden1 = crbm1.getHidden(data, crbm1DataEdgeLength);

        //float[][][] maxPooled1 = maxPooling(hidden1, crbm1PoolingSize, crbm1DataEdgeLength, crbm1FilterEdgeLength);
        //int crbm2DataEdgeLength = maxPoolEdgeCalc(crbm1PoolingSize, crbm1DataEdgeLength, crbm1FilterEdgeLength);

        // EXPORT
        exportAsImage(reduceDimension(hidden1), "hidden1");
        // EXPORT END

        CRBM crbm2 = new CRBM(K, crbmFilterEdgeLength);
        crbm2.train(reduceDimension(hidden1), crbm2DataEdgeLength, epochs, learningRate, "Second-RBM");
        crbm2.killFirst();
        float[][][] hidden2 = crbm2.getHidden(reduceDimension(hidden1), crbm2DataEdgeLength);

        // EXPORT
        exportAsImage(reduceDimension(hidden2), "hidden2");
        // EXPORT END

        float[][] visible2 = crbm2.getVisible(hidden2, null, crbm2DataEdgeLength);

        // EXPORT
        exportAsImage(visible2, "visible2");
        // EXPORT END

        float[][] visible1 = crbm1.getVisible(expandDimension(visible2, K), null, crbm2DataEdgeLength - crbmFilterEdgeLength + 1);

        // EXPORT
        exportAsImage(visible1, "visible1");
        // EXPORT END

        //float[][][] nn = nearestNeighbour(expandDimension(visible2, K), crbm2DataEdgeLength - crbm2FilterOffset, crbm2DataEdgeLength - crbm2FilterOffset, (crbm2DataEdgeLength - crbm2FilterOffset) * crbm1PoolingSize, (crbm2DataEdgeLength - crbm2FilterOffset) * crbm1PoolingSize);

        //int crbm1FilterOffset = crbm1FilterEdgeLength - 1;
        //float[][] visible = crbm1.getVisible(nn, null, (crbm2DataEdgeLength - crbm2FilterOffset) * crbm1PoolingSize - crbm1FilterOffset);

        //float[][][] maxPooled2 = maxPooling(hidden2, crbm2PoolingSize, crbm2DataEdgeLength, crbm2FilterEdgeLength);

        //exportAsImage(visible, "visible");

    }

    float[][][] expandDimension(float[][] data, int nextDimensionSize) {
        float[][][] result = new float[data.length / nextDimensionSize][nextDimensionSize][data[0].length];

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                for(int k = 0; k < result[0][0].length; k++) {
                    result[i][j][k] = data[i * result[0].length + j][k];
                }
            }
        }
        return result;
    }

    float[][] reduceDimension(float[][][] data) {
        float[][] result = new float[data.length * data[0].length][];

        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < data[0].length; j++){
                result[i * data[0].length + j] = data[i][j];
            }
        }
        return result;
    }

    public float[][][] nearestNeighbour(float[][][] pixels, int actualWidth, int actualHeight, int newWidth, int newHeight) {
        float[][][] result = new float[pixels.length][pixels[0].length][];
        for(int i = 0; i < pixels.length; i++) {
            result[i] = nearestNeighbour(pixels[i], actualWidth, actualHeight, newWidth, newHeight);
        }

        return result;
    }

    public float[][] nearestNeighbour(float[][] pixels, int actualWidth, int actualHeight, int newWidth, int newHeight) {
        float[][] result = new float[pixels.length][];

        for(int i = 0; i < pixels.length; i++) {
            result[i] = nearestNeighbour(pixels[i], actualWidth, actualHeight, newWidth, newHeight);
        }

        return result;
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

    private float[][][] maxPooling(float[][][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        float[][][] result = new float[data.length][K][];

        for(int i = 0; i < data.length; i++) {
            result[i] = maxPooling(data[i], poolingSize, dataEdgeLength , filterEdgeLength);
        }

        return result;
    }

//    private float[][] maxPooling(float[][][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
//        float[][] result = new float[data.length * K][];
//
//        for(int i = 0; i < data.length; i++) {
//            float[][] r = maxPooling(data[i], poolingSize, dataEdgeLength , filterEdgeLength);
//            for(int j = 0; j < r.length; j++){
//                result[K * i + j] = r[j];
//            }
//        }
//
//        return result;
//    }

    private int maxPoolEdgeCalc(int poolingSize, int dataEdgeLength, int filterEdgeLength){

        int offset = filterEdgeLength-1;
        int pEdgeLength = dataEdgeLength - offset;

        if(pEdgeLength % poolingSize != 0){
            pEdgeLength = (pEdgeLength / poolingSize + 1) * poolingSize;

        }
        return pEdgeLength / poolingSize;
    }

    private float[][] maxPooling(float[][] data, int poolingSize, int dataEdgeLength ,int filterEdgeLength) {
        int offset = filterEdgeLength-1;
        int pEdgeLength = dataEdgeLength - offset;
        int iEdgeLength = dataEdgeLength - offset;

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

                            max = Math.max(value, max);

                        }
                    }

                    result[k][y * rEdgeLength + x] = max;
                }
            }
        }

        return result;
    }

    private void exportAsImage(float[][] data, String name) {

        String exportPath = "export";


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
