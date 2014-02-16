package crbm;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by Moritz on 2/16/14.
 */
public class ConfusionMatrix {
    private final ArrayList<String> classes;
    private final int[][] table;

    public ConfusionMatrix(Collection<String> classes) {
        this.classes = new ArrayList<String>(classes);
        this.table = new int[classes.size()][classes.size()];
    }


    public void addEntry(String actualClass, String predictedClass) {
        if(classes.contains(actualClass) && classes.contains(predictedClass)) {
            int actualEntry = classes.indexOf(actualClass);
            int predictedEntry = classes.indexOf(predictedClass);
            table[actualEntry][predictedEntry]++;
        }
    }

    public int[][] getTable() {
        return table;
    }
}
