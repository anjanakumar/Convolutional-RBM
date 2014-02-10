/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package crbm;

/**
 *
 * @author christoph
 */
public class DataSet {
    private final float[] data;
    private final String label;
    
    public DataSet(float[] data, String label){
        this.data = data;
        this.label = label;
    }
    
    public float[] getData(){
        return data;
    }
    
    public String getLabel(){
        return label;
    }
}
