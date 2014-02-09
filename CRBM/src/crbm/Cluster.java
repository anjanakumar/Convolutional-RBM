/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package crbm;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 *
 * @author christoph
 */
public class Cluster {
    
    private final List<float[]> data;    
    private float[] center;
    private float totalDistance;
    
    public Cluster(float[][] data){
        this.data = Arrays.asList(data);
        init();
    }
    
    public final void init(){
        this.center = center();
        this.totalDistance = totalDistance();
    }
    
    public Cluster(float[] center){
        this.data = null;//new LinkedList<>();
        this.center = center;
        this.totalDistance = 0f;
    }
    
    private float[] center(){
        if(data.isEmpty()) return null;
        
        int size = data.size();
        int len = data.get(0).length;
        
        float[] result = new float[len];
        
        for(float[] v : data){
            for(int i = 0; i < v.length; ++i){
                center[i] += v[i];
            }
        }
        
        for(int i = 0; i < len; ++i){
            result[i] /= size;
        }
        return result;
    }
    
    private float totalDistance() {
        if(center == null) return 0;
        
        float sum = 0f;
        for(float[] v : data){
            sum += distanceToCenter(v);
        }
        return sum;
    }
    
    public void addVector(float[] v){
        this.data.add(v);
    }
    
    public List<float[]> getData(){
        return data;
    }
    
    public float[] getCenter(){
        return center;
    }
    
    public float getTotalDistance(){
        return totalDistance;
    }
    
    public float distanceToCenter(float[] v){
        float sum = 0f;
        for(int i = 0; i < center.length; ++i){
            sum += (center[i] - v[i]) * (center[i] - v[i]);
        }
        return (float) Math.sqrt(sum);
    }
}
