package crbm;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

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



}
