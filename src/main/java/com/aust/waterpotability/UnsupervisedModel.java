package com.aust.waterpotability;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class UnsupervisedModel {
    public static void main(String[] args) throws Exception {
        Instances data = DataSource.read("data/processed_water_potability.arff");
        data.setClassIndex(data.numAttributes() - 1);

        // Apply k-Means with 2 clusters (potable vs not potable)
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(2);
        kmeans.buildClusterer(data);

        System.out.println("===== KMeans Clustering =====");
        System.out.println(kmeans);
    }
}
